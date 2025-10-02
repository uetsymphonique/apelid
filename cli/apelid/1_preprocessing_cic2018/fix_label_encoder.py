import os
import sys
from sklearn.preprocessing import LabelEncoder
import joblib

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from configs.cic2018 import CIC2018Resources


# load from encoders/cic2018/label_encoder.pkl
label_encoder = LabelEncoder()
label_encoder = joblib.load(f'encoders/cic2018/label_encoder.pkl')

# print the class names
print(label_encoder.classes_)

new_class_names = CIC2018Resources.MAJORITY_LABELS + CIC2018Resources.MINORITY_LABELS

new_label_encoder = LabelEncoder()

# fit the new label encoder
new_label_encoder.fit(new_class_names)

print(new_label_encoder.classes_)

# dump the new label encoder
joblib.dump(new_label_encoder, f'encoders/cic2018/label_encoder_fixed.pkl')

