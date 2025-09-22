echo "Setup encoders: [run.sh]$ python -m cli.cic2018.preprocessing.setup_encoders"
python -m cli.cic2018.preprocessing.setup_encoders

echo "Encode: [run.sh]$ python -m cli.cic2018.major.1_encode.encode --subset full"
python -m cli.cic2018.major.1_encode.encode --subset full

echo "Fit PCA: [run.sh]$ python -m cli.cic2018.major.1_encode.pca_fit"
python -m cli.cic2018.major.1_encode.pca_fit

echo "Transform PCA: [run.sh]$ python -m cli.cic2018.major.1_encode.pca_transform --subset train/test"
python -m cli.cic2018.major.1_encode.pca_transform --subset train --log-level WARNING
python -m cli.cic2018.major.1_encode.pca_transform --subset test --log-level WARNING

echo "KMeans compression: [run.sh]$ python -m cli.cic2018.major.kmeans_compression.kmeans_compression --subset train --label All --budget 18800 --log-level DEBUG"
python -m cli.cic2018.major.kmeans_compression.kmeans_compression --subset train --label All --budget 75200 --log-level DEBUG
echo "KMeans compression: [run.sh]$ python -m cli.cic2018.major.kmeans_compression.kmeans_compression --subset test --label All --budget 18800 --log-level DEBUG"
python -m cli.cic2018.major.kmeans_compression.kmeans_compression --subset test --label All --budget 18800 --log-level DEBUG


echo "Finalize decode: [run.sh]$ python -m cli.cic2018.major.kmeans_compression.finalize_decode --subset train/test --label All --log-level WARNING"
python -m cli.cic2018.major.kmeans_compression.finalize_decode --subset train --label All --log-level WARNING
python -m cli.cic2018.major.kmeans_compression.finalize_decode --subset test --label All --log-level WARNING


echo "Minor encode: [run.sh]$ python -m cli.cic2018.minor.encode --subset full --log-level INFO"
python -m cli.cic2018.minor.encode --subset full --log-level INFO

echo "Minor augment: [run.sh]$ python -m cli.cic2018.minor.augmenting --augmenting-strategy wgan --mode all --tau 75200 --log-level INFO"
python -m cli.cic2018.minor.augmenting --augmenting-strategy wgan --mode all --tau 75200 --log-level INFO

echo "Minor decode: [run.sh]$ python -m cli.cic2018.minor.decode_augmented --strategy wgan --mode all --log-level INFO"
python -m cli.cic2018.minor.decode_augmented --strategy wgan --mode all --log-level INFO


echo "Merge major minor: [run.sh]$ python -m cli.cic2018.preparing.merge_major_minor --strategy wgan"
python -m cli.cic2018.preparing.merge_major_minor --strategy wgan
python -m cli.cic2018.preparing.prepare_data_for_train_phase --log-level WARNING

echo "training:"
python -m cli.cic2018.training.catboost --cat-cols Protocol "Dst Port" 'RST Flag Cnt' 'PSH Flag Cnt' 'ACK Flag Cnt' 'ECE Flag Cnt'  --log-level DEBUG
python -m cli.cic2018.training.catboost --log-level DEBUG

