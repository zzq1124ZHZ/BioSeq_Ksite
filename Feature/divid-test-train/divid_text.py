import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv( r"concreat_neg_feature.csv",header=None)
# train and test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=3047)
test_df.to_csv(r'divid_test_train\test_dataset.csv', index=False)
train_df.to_csv(r'divid_test_train\train_dataset.csv', index=False)
