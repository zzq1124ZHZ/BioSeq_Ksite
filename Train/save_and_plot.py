import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve,matthews_corrcoef
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

def save_and_plot(arr_probs, arr_labels):
    data = pd.DataFrame({
        'Probabilities': arr_probs,
        'Labels': arr_labels
    })
    data.to_csv('Stack.csv', index=False)
    return data
