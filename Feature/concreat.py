#
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

meiler_train_neg_40_feature= pd.read_csv(r"meiler_neg.csv",header=None)#11800
pt5_train_neg_40_feature= pd.read_csv(r"pt5_negative.csv",header=None)
print(BLOSUM62_train_neg_40_feature.shape)#(9344, 50)
print(pt5_train_neg_40_feature.shape)#(9344, 1024)
concreat_feature_neg=np.concatenate([pt5_train_neg_40_feature,meiler_train_neg_40_feature], axis=1)
# 将结果保存到CSV文件
csv_file_path= r'concreat_neg_feature.csv'#
np.savetxt(csv_file_path,concreat_feature_neg, delimiter=',')
print(f"CSV文件已保存到: {csv_file_path}")

# meiler_train_pos_40_feature= pd.read_csv(r"meiler_neg.csv",header=None)#11800
# pt5_train_pos_40_feature= pd.read_csv(r"pt5_negative.csv",header=None)
# print(BLOSUM62_train_pos_40_feature.shape)#(9344, 50)
# print(pt5_train_pos_40_feature.shape)#(9344, 1024)
# concreat_feature_pos=np.concatenate([pt5_train_pos_40_feature,meiler_train_pos_40_feature], axis=1)
# #
# # 将结果保存到CSV文件
# csv_file_path= r'concreat_neg_feature.csv'#
# np.savetxt(csv_file_path,concreat_feature_pos, delimiter=',')
# print(f"CSV文件已保存到: {csv_file_path}")