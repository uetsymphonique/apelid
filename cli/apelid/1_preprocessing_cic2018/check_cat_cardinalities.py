import joblib
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

cic2018_ordinal_encoder = joblib.load('encoders/cic2018/ordinal_encoder.pkl')

for feature, category in zip(cic2018_ordinal_encoder.feature_names_in_, cic2018_ordinal_encoder.categories_):
    print(feature, len(category))



nslkdd_ordinal_encoder = joblib.load('encoders/nslkdd/ordinal_encoder.pkl')

for feature, category in zip(nslkdd_ordinal_encoder.feature_names_in_, nslkdd_ordinal_encoder.categories_):
    print(feature, len(category))