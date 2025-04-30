from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

meiler_feature= pd.read_csv(r"meiler.csv",header=None)
pt5_feature= pd.read_csv(r"pt5.csv",header=None)
concreat_feature_neg=np.concatenate([pt5_feature,meiler_feature], axis=1)
csv_file_path= r'feature.csv'
np.savetxt(csv_file_path,concreat_feature_neg, delimiter=',')

