import os
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

import numpy as np
import pandas as pd

from utils.logging import get_logger
from resampling.data_augmentation.augmented_wgan.wgan import WGAN
from eval.postfilter import PostFilterClassifier


logger = get_logger(__name__)


def encode_with_preprocessor(pre, df: pd.DataFrame) -> pd.DataFrame:
    df_enc = pre.preprocess_encode_numerical_features(df.copy())
    df_enc = pre.preprocess_encode_binary_features(df_enc)
    df_enc = pre.preprocess_encode_label(df_enc)
    df_enc = pre.preprocess_encode_categorical_features(df_enc)
    return df_enc


def build_encoded_keys(df_enc: pd.DataFrame, feat_cols: Iterable[str], decimals: int = 6) -> set:
    arr = df_enc[list(feat_cols)].astype(float).round(decimals).values
    return set(map(tuple, arr))


def train_wgan_with_critic(encoded_train: pd.DataFrame,
                           feat_cols: Iterable[str],
                           pre,
                           benign_encoded: Optional[pd.DataFrame],
                           device: str = 'auto',
                           use_gp: bool = True,
                           critic_epochs: int = 60,
                           wgan_iterations: int = 10000,
                           d_iter: int = 5) -> WGAN:
    x_dim = len(list(feat_cols))
    wgan = WGAN(x_dim=x_dim, device=device, use_gp=use_gp, use_critic_loss=True, lambda_critic=0.5)

    # Critic train set
    if benign_encoded is not None and not benign_encoded.empty:
        pos = encoded_train.copy()
        neg = benign_encoded.sample(n=max(len(pos) * 4, 1), replace=True, random_state=1)
        critic_df = pd.concat([pos, neg], ignore_index=True).sample(frac=1, random_state=1)
        logger.info(f"[+] Critic set: pos={len(pos)}, neg={len(neg)}")
    else:
        critic_df = encoded_train.copy()

    critic_loader, _ = wgan.prepare_data(critic_df, use_label_column=True)
    wgan.train_critic(critic_loader, epochs=critic_epochs)

    attack_loader, _ = wgan.prepare_data(encoded_train, use_label_column=False)
    wgan.train_wgan(attack_loader, iterations=wgan_iterations, d_iter=d_iter, save_interval=1000)
    return wgan


@dataclass
class AugmentOptions:
    # Data/critic
    use_benign_for_critic: bool = True
    critic_epochs: int = 60
    wgan_iterations: int = 10000
    d_iter: int = 5
    use_gp: bool = True

    # Generation/selection
    accept_rate: float = 0.2
    request_multiplier: float = 3.0
    max_rounds: int = 40

    # Post-filtering
    use_postfilter: bool = True
    min_precision: float = 0.95

    # Dedup/trim/fill
    use_encoded_dedup: bool = True
    trim_to_need: bool = True
    use_final_fill: bool = True


def _build_critic_df(encoded_train: pd.DataFrame, benign_encoded: Optional[pd.DataFrame]) -> pd.DataFrame:
    if benign_encoded is not None and not benign_encoded.empty:
        pos = encoded_train.copy()
        neg = benign_encoded.sample(n=max(len(pos) * 4, 1), replace=True, random_state=1)
        return pd.concat([pos, neg], ignore_index=True).sample(frac=1, random_state=1)
    return encoded_train.copy()


def _train_postfilter(pos_X: pd.DataFrame, benign_X: Optional[pd.DataFrame], min_precision: float) -> PostFilterClassifier:
    if benign_X is None or benign_X.empty:
        benign_X = pos_X.sample(n=len(pos_X), replace=True, random_state=2)
    pos_y = np.ones(len(pos_X), dtype=np.int32)
    neg_y = np.zeros(len(benign_X), dtype=np.int32)
    X_all = pd.concat([pos_X, benign_X], ignore_index=True)
    y_all = np.concatenate([pos_y, neg_y])
    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_tr, y_val = train_test_split(X_all, y_all, test_size=0.2, random_state=123, stratify=y_all)
    clf = PostFilterClassifier().fit(X_tr, y_tr)
    clf.calibrate_threshold(X_val, y_val, min_precision=min_precision)
    return clf


def _postfilter_candidates(clf: PostFilterClassifier, gen_df: pd.DataFrame, need_now: int) -> pd.DataFrame:
    scores = clf.predict_proba(gen_df.drop(columns=['Label']))
    if getattr(clf, 'threshold', None) is not None:
        keep_mask = scores >= clf.threshold
        kept = int(keep_mask.sum())
        if kept == 0:
            topk_idx = np.argsort(-scores)[:min(len(scores), need_now)]
            return gen_df.iloc[topk_idx]
        return gen_df.loc[keep_mask]
    # No threshold calibrated, fallback top-k
    topk_idx = np.argsort(-scores)[:min(len(scores), need_now)]
    return gen_df.iloc[topk_idx]


def _dedup_encoded(gen_df: pd.DataFrame, feat_cols: Iterable[str], real_keys: set, accepted_batches: list[pd.DataFrame]) -> pd.DataFrame:
    already = real_keys.copy()
    for b in accepted_batches:
        already |= build_encoded_keys(b, feat_cols)
    unique_mask = [tuple(row) not in already for row in gen_df[list(feat_cols)].astype(float).round(6).values]
    return gen_df.loc[unique_mask]


def _trim_to_need(accepted_enc: pd.DataFrame, need: int, clf: Optional[PostFilterClassifier]) -> pd.DataFrame:
    if len(accepted_enc) <= need:
        return accepted_enc
    try:
        if clf is not None:
            scores_all = clf.predict_proba(accepted_enc.drop(columns=['Label']))
            top_idx = np.argsort(-scores_all)[:need]
            return accepted_enc.iloc[top_idx]
        return accepted_enc.sample(n=need, random_state=42)
    except Exception:
        return accepted_enc.head(need)


def _final_fill(pre,
                wgan: WGAN,
                class_name: str,
                enc_train: pd.DataFrame,
                benign_enc: Optional[pd.DataFrame],
                feat_cols: Iterable[str],
                real_keys: set,
                accepted_enc: pd.DataFrame,
                deficit: int,
                accept_rate: float) -> pd.DataFrame:
    pos_X = enc_train.drop(columns=['Label'])
    if benign_enc is not None and not benign_enc.empty:
        neg_X = benign_enc.drop(columns=['Label']).sample(n=min(len(pos_X)*4, len(benign_enc)), replace=True, random_state=7)
    else:
        neg_X = pos_X.sample(n=len(pos_X), replace=True, random_state=7)
    from sklearn.model_selection import train_test_split
    X_all = pd.concat([pos_X, neg_X], ignore_index=True)
    y_all = np.concatenate([np.ones(len(pos_X), dtype=np.int32), np.zeros(len(neg_X), dtype=np.int32)])
    X_tr, _, y_tr, _ = train_test_split(X_all, y_all, test_size=0.2, random_state=7, stratify=y_all)
    clf_fill = PostFilterClassifier().fit(X_tr, y_tr)
    request = int(min(max(deficit * 6, (deficit / max(accept_rate, 1e-3)) * 6), 80000))
    gen_extra = wgan.generate_samples(request, critic_threshold=None, accept_rate=accept_rate)
    gen_extra_df = pd.DataFrame(gen_extra, columns=list(feat_cols))
    gen_extra_df['Label'] = enc_train['Label'].iloc[0]
    scores = clf_fill.predict_proba(gen_extra_df.drop(columns=['Label']))
    gen_extra_df = gen_extra_df.iloc[np.argsort(-scores)[:min(len(scores), deficit)]]
    already = real_keys | build_encoded_keys(accepted_enc, feat_cols)
    unique_mask = [tuple(row) not in already for row in gen_extra_df[list(feat_cols)].astype(float).round(6).values]
    gen_extra_unique = gen_extra_df.loc[unique_mask]
    if len(gen_extra_unique) > 0:
        accepted_enc = pd.concat([accepted_enc, gen_extra_unique], ignore_index=True)
    return accepted_enc


def generate_augmented_samples(pre,
                               class_name: str,
                               train_df: pd.DataFrame,
                               test_df: pd.DataFrame,
                               benign_loader: Optional[Callable[[], Optional[pd.DataFrame]]],
                               tau: int = 14000,
                               device: str = 'auto',
                               accept_rate: float = 0.2,
                               min_precision: float = 0.95,
                               options: Optional[AugmentOptions] = None) -> pd.DataFrame:
    """Augment a minority class to tau and RETURN ENCODED DATAFRAME (no inverse).

    Flow: encode → train critic/WGAN → generate encoded → optional postfilter → encoded-dedup →
    optional fill/trim → return enc_train + accepted_enc.
    """
    opts = options or AugmentOptions(accept_rate=accept_rate, min_precision=min_precision)
    logger.info(f"[+] Augmenting {class_name}: {len(train_df)} -> {tau}")
    need = tau - len(train_df)
    if need <= 0:
        logger.info(f"[+] {class_name} already >= tau")
        return train_df

    # Encode
    enc_train = encode_with_preprocessor(pre, train_df)
    enc_test = encode_with_preprocessor(pre, test_df)
    feat_cols = [c for c in enc_train.columns if c != 'Label']

    # Load benign encoded (optional)
    benign_enc = benign_loader() if (benign_loader is not None and opts.use_benign_for_critic) else None

    # Train WGAN + critic
    wgan = train_wgan_with_critic(
        enc_train, feat_cols, pre, benign_enc, device=device,
        use_gp=opts.use_gp, critic_epochs=opts.critic_epochs,
        wgan_iterations=opts.wgan_iterations, d_iter=opts.d_iter,
    )

    # Save artifacts
    safe_name = class_name.lower().replace(' ', '_').replace('/', '_')
    wgan.save_models(f"models/wgan_cic_{safe_name}")
    os.makedirs("wgan_losses", exist_ok=True)
    wgan.plot_losses(f"wgan_losses/wgan_losses_cic_{safe_name}.png")

    # Dedup keys
    real_keys = build_encoded_keys(enc_train, feat_cols) | build_encoded_keys(enc_test, feat_cols)
    accepted_batches = []
    last_clf: Optional[PostFilterClassifier] = None

    # Loop generate → post-filter → dedup until enough
    rounds = 0
    while sum(len(b) for b in accepted_batches) < need and rounds < opts.max_rounds:
        rounds += 1
        need_now = need - sum(len(b) for b in accepted_batches)
        request = int(min(max(need_now * opts.request_multiplier, (need_now / max(opts.accept_rate, 1e-3)) * opts.request_multiplier), 60000))
        logger.info(f"[+] Round {rounds}/{opts.max_rounds}: need_now={need_now}, request={request}, accept_rate={opts.accept_rate}")
        gen = wgan.generate_samples(request, critic_threshold=None, accept_rate=opts.accept_rate)
        gen_df = pd.DataFrame(gen, columns=feat_cols)
        class_id = pre.encoders['label'].transform([class_name])[0]
        gen_df['Label'] = class_id

        # Post-filter training (minor vs benign) – optional
        if opts.use_postfilter:
            try:
                pos_X = enc_train.drop(columns=['Label'])
                benign_X = benign_enc.drop(columns=['Label']) if (benign_enc is not None and not benign_enc.empty) else None
                clf = _train_postfilter(pos_X, benign_X, min_precision=opts.min_precision)
                last_clf = clf
                gen_df = _postfilter_candidates(clf, gen_df, need_now)
            except Exception as e:
                logger.warning(f"[PostFilter] skipped: {e}")

        # Dedup encoded vs real and already accepted – optional (enabled by default)
        gen_unique = gen_df
        if opts.use_encoded_dedup:
            gen_unique = _dedup_encoded(gen_df, feat_cols, real_keys, accepted_batches)
        if len(gen_unique) > 0:
            accepted_batches.append(gen_unique)

    accepted_enc = pd.concat(accepted_batches, ignore_index=True) if accepted_batches else pd.DataFrame(columns=feat_cols+['Label'])
    if opts.trim_to_need:
        accepted_enc = _trim_to_need(accepted_enc, need, last_clf)

    # Final fill (optional) if still short
    if opts.use_final_fill:
        deficit = need - len(accepted_enc)
        if deficit > 0:
            try:
                accepted_enc = _final_fill(pre, wgan, class_name, enc_train, benign_enc, feat_cols, real_keys, accepted_enc, deficit, opts.accept_rate)
                if opts.trim_to_need:
                    accepted_enc = _trim_to_need(accepted_enc, need, last_clf)
            except Exception as e:
                logger.warning(f"[Fill] skipped: {e}")

    # Return encoded augmented dataframe (enc_train + accepted_enc)
    augmented_enc = pd.concat([enc_train, accepted_enc], ignore_index=True)
    logger.info(f"[+] {class_name} augmented (encoded): {len(enc_train)} + {len(accepted_enc)} -> {len(augmented_enc)}")
    return augmented_enc


