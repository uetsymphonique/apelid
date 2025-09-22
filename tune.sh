# python -m cli.cic2018.major.3_confusion_handle.compute_boundary --target Benign --neighbor Infilteration --reuse-existing  --budget-total 1750 --margin-low 0.9 --margin-high 1.2 --use-relative-margin --select --log-level DEBUG
# python -m cli.cic2018.major.3_confusion_handle.coreset_train --label Benign --use-relative-margin --log-level INFO --budget-total 70000
# python -m cli.cic2018.major.5_complete.finalize_decode --subset train --label "Benign" --log-level WARNING

# # python -m cli.cic2018.major.3_confusion_handle.compute_boundary --neighbor Benign --target Infilteration --reuse-existing  --budget-total 35000 --margin-low 0.75 --margin-high 1.45 --use-relative-margin --select --log-level DEBUG
# python -m cli.cic2018.major.3_confusion_handle.coreset_train --label Infilteration --use-relative-margin --log-level INFO --overlap-ratio 0.75 --budget-total 70000
# python -m cli.cic2018.major.5_complete.finalize_decode --subset train --label "Infilteration" --log-level WARNING

python -m cli.cic2018.preparing.merge_major_minor --strategy wgan
python -m cli.cic2018.preparing.prepare_data_for_train_phase
python -m cli.cic2018.training.catboost --cat-cols Protocol "Dst Port" --log-level DEBUG