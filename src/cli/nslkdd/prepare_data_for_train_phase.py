import pandas as pd
from preprocessing.nslkdd_preprocessor import NSLKDDPreprocessor
from dataservice.data_service import DataService

nslkdd_preprocessor = NSLKDDPreprocessor()
nslkdd_preprocessor.load_encoders()


train_df = pd.read_csv('data/final_train_balanced.csv')
print(train_df.shape)
test_df = pd.read_csv('data/final_test.csv')
print(test_df.shape)

train_df = nslkdd_preprocessor.preprocess_encode_ordinal_features(train_df)
test_df = nslkdd_preprocessor.preprocess_encode_ordinal_features(test_df)

# train_df = nslkdd_preprocessor.preprocess_encode_label(train_df)
# test_df = nslkdd_preprocessor.preprocess_encode_label(test_df)
print(train_df.head())
print(test_df.head())

train_df_svc = DataService(train_df)
test_df_svc = DataService(test_df)

train_df_svc.fix_duplicates()
test_df_svc.fix_duplicates()

categorical_features = ['protocol_type', 'service', 'flag', 'land', 'logged_in', 'is_host_login', 'is_guest_login']
df = pd.concat([train_df, test_df])
print("categorical features details (merged train and test):")
for col in categorical_features:
    # number of unique values in the column
    print(f"{col}: {df[col].nunique()}")
    # min and max value in the column
    print(f"{col}: min={df[col].min()}, max={df[col].max()}")
    

train_df_svc.export_data('data/KDD+_final_train_balanced_cat_map.csv')
test_df_svc.export_data('data/KDD+_final_test_cat_map.csv')





