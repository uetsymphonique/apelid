python -m cli.cic2018.major.tsne_on_embeddings --subset train --components Benign Infilteration --log-level WARNING --save-png
python -m cli.cic2018.major.tsne_on_embeddings --subset train --components Infilteration Benign --log-level WARNING --save-png
python -m cli.cic2018.major.tsne_on_embeddings --subset train --components Benign Boundary_Benign_to_Infil --use-coreset --exclude-boundary-overlap --log-level WARNING --save-png
python -m cli.cic2018.major.tsne_on_embeddings --subset train --components Benign Boundary_Benign_to_Infil --exclude-boundary-overlap --log-level WARNING --save-png
python -m cli.cic2018.major.tsne_on_embeddings --subset train --components Infilteration Boundary_Infil_to_Benign --use-coreset --exclude-boundary-overlap --log-level DEBUG --save-png
python -m cli.cic2018.major.tsne_on_embeddings --subset train --components Infilteration Boundary_Infil_to_Benign --exclude-boundary-overlap --log-level WARNING --save-png
python -m cli.cic2018.major.tsne_on_embeddings --subset train --components Benign Boundary_Benign_to_Infil Infilteration Boundary_Infil_to_Benign --use-coreset --exclude-boundary-overlap --log-level DEBUG --save-png
python -m cli.cic2018.major.tsne_on_embeddings --subset train --components Infilteration Boundary_Infil_to_Benign Benign Boundary_Benign_to_Infil --use-coreset --exclude-boundary-overlap --log-level WARNING --save-png
python -m cli.cic2018.major.tsne_on_embeddings --subset train --components Infilteration Benign --use-coreset --log-level WARNING --save-png
