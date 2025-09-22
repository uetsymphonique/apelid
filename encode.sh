echo "Merge major minor: [run.sh]$ python -m cli.cic2018.preparing.merge_major_minor --strategy wgan"
python -m cli.cic2018.preparing.merge_major_minor --strategy wgan
# python -m cli.cic2018.preparing.prepare_data_for_train_phase --log-level WARNING
python -m cli.cic2018.preparing.prepare_data_for_test_phase --log-level WARNING

echo "training:"
# python -m cli.cic2018.training.catboost --cat-cols Protocol "Dst Port" 'RST Flag Cnt' 'PSH Flag Cnt' 'ACK Flag Cnt' 'ECE Flag Cnt'  --log-level DEBUG
python -m cli.cic2018.training.catboost --log-level DEBUG