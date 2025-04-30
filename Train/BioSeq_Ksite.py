import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
import random
from save_and_plot import save_and_plot

class CrossAttention(nn.Module):
    def __init__(self, input_dim, qk_dim, v_dim):
        super(CrossAttention, self).__init__()
        self.q = nn.Linear(input_dim, qk_dim)
        self.cn6 = nn.Conv1d(input_dim, qk_dim, kernel_size=6, padding='same', groups=input_dim) 
        self.pointwise_conv = nn.Conv1d(qk_dim, qk_dim, kernel_size=1) 
        self._norm_fact = 1 / torch.sqrt(torch.tensor(qk_dim, dtype=torch.float32))
        self.learned_positional_encoding = None  

    def forward(self, evo_local):
        batch_size, seq_len, _ = evo_local.size() 
        if self.learned_positional_encoding is None:
            self.learned_positional_encoding = nn.Parameter(torch.randn(1, seq_len, self.q.out_features)).to(
                evo_local.device) 
        Q = self.q(evo_local)
        Q = Q + self.learned_positional_encoding  
        k_v_features = self.cn6(evo_local.permute(0, 2, 1))  # Shape: (batch_size, qk_dim, seq_len)
        k_v_features = self.pointwise_conv(k_v_features)  # Shape: (batch_size, qk_dim, seq_len)
        K = k_v_features.permute(0, 2, 1)  # Shape: (batch_size, seq_len, qk_dim)
        V = k_v_features.permute(0, 2, 1)  # Shape: (batch_size, seq_len, v_dim)
        atten_scores = torch.bmm(Q, K.permute(0, 2, 1)) * self._norm_fact 
        atten = F.softmax(atten_scores, dim=-1)
        output = torch.bmm(atten, V)
        return output + V + evo_local

class CustomEncoder(nn.Module):
    def __init__(self, input_dim, d_model):
        super(CustomEncoder, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, d_model, kernel_size=6, padding='same')
        self.bi_lstm = nn.LSTM(d_model, d_model, batch_first=True, bidirectional=True)
        self.cross_attention1 = CrossAttention(input_dim=2 * d_model, qk_dim=2 * d_model, v_dim=2 * d_model)
        self.cross_attention2 = CrossAttention(input_dim=2 * d_model, qk_dim=2 * d_model, v_dim=2 * d_model)
        self.gate_layer = nn.Linear(2 * d_model, 2 * d_model)
    def forward(self, src):
        src_conv = src.permute(0, 2, 1)  # Shape: (batch_size, input_dim, seq_len)
        src_conv = self.conv1(src_conv)  # Shape: (batch_size, d_model, seq_len)
        src_conv = src_conv.permute(0, 2, 1)  # Shape: (batch_size, seq_len, d_model)
        lstm_out, _ = self.bi_lstm(src_conv)  # Shape: (batch_size, seq_len, 2 * d_model)
        branch1_output = self.cross_attention1(lstm_out)
        branch2_output = self.cross_attention2(lstm_out)
        gate = torch.sigmoid(self.gate_layer(branch1_output + branch2_output))
        combined_output = gate * branch1_output + (1 - gate) * branch2_output
        return combined_output

class Seq2SeqModel(nn.Module):
    def __init__(self, input_dim, d_model):
        super(Seq2SeqModel, self).__init__()
        self.encoder = CustomEncoder(input_dim, d_model)
        self.flatten = nn.Flatten()
        output_length = 1255  
        self.fc1 = nn.Linear(2 * d_model * output_length, d_model)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(d_model, d_model // 2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(d_model // 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, src):
        enc_out = self.encoder(src)
        flattened = self.flatten(enc_out)
        x = self.fc1(flattened)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        final_output = self.fc3(x)
        final_output = self.sigmoid(final_output)
        return final_output
    
class MyDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32).permute(0, 2, 1) 
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
      
pos_feature_path = r"divid_test_train\train_dataset_pos_2.csv"
neg_feature_path = r"divid_test_train\train_dataset_neg_2.csv"
pos_features = pd.read_csv(pos_feature_path)
neg_features = pd.read_csv(neg_feature_path)
X = np.vstack((pos_features, neg_features))
y = np.concatenate((np.ones(pos_features.shape[0]), np.zeros(neg_features.shape[0])))
x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=seed)

pos_features_test = pd.read_csv(r"divid_test_train\test_dataset_pos_2.csv")#
neg_features_test = pd.read_csv(r"divid_test_train\test_dataset_neg_2.csv")
X_test = np.vstack((pos_features_test, neg_features_test))
y_test = np.concatenate((np.ones(pos_features_test.shape[0]), np.zeros(neg_features_test.shape[0])))

x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])  # (batch_size, 1, num_features)
x_val = x_val.reshape(x_val.shape[0], 1, x_val.shape[1])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

train_dataset = MyDataset(x_train, y_train)
val_dataset = MyDataset(x_val, y_val)
test_dataset = MyDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=32)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for features, labels in val_loader:
            features = features.to(device)
            labels = labels.to(device)
            outputs = model(features)  
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    val_loss = total_loss / len(val_loader)
    return val_loss


def train(model, train_loader, val_loader, device, optimizer, criterion, epochs=100):
    print(f"Using device: {device}")

    early_stopping = EarlyStopping(patience=5, verbose=True, path='best_model.pth')
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch_idx, (features, labels) in enumerate(train_loader):
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(features) 
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        val_loss = validate(model, val_loader, criterion, device)
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {total_loss / len(train_loader):.4f}, Validation Loss: {val_loss:.4f}')
    print('Finished Training')


def test(model, test_loader, device):
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    arr_probs = []
    arr_labels = []
    arr_labels_hyps = []
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            labels = labels.to(device)
            y_pred = model(features)
            probs = y_pred.squeeze(1).cpu().numpy().flatten()
            labels_np = labels.cpu().numpy().flatten()
            y_pred_labels = (y_pred > 0.5).int().cpu().numpy().flatten()
            arr_probs.extend(probs.tolist())
            arr_labels.extend(labels_np.tolist())
            arr_labels_hyps.extend(y_pred_labels.tolist())

    tn, fp, fn, tp = metrics.confusion_matrix(arr_labels, arr_labels_hyps).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    f1score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    metrics_dict = {
        'accuracy': metrics.accuracy_score(arr_labels, arr_labels_hyps),
        'balanced_accuracy': metrics.balanced_accuracy_score(arr_labels, arr_labels_hyps),
        'MCC': metrics.matthews_corrcoef(arr_labels, arr_labels_hyps),
        'AUC': metrics.roc_auc_score(arr_labels, arr_probs),
        'AP': metrics.average_precision_score(arr_labels, arr_probs),
        'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1score,
        'Youden Index': sensitivity + specificity - 1
    }
    df = pd.DataFrame([metrics_dict])
    df.to_csv('test_results.csv', index=False)
    print('Test results saved to CSV file.')
    for key, value in metrics_dict.items():
        print(f'{key}: {value}')
    save_and_plot(arr_probs, arr_labels)
    
    return metrics_dict['accuracy'], metrics_dict['MCC']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Seq2SeqModel(input_dim=1, d_model=14).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)


class SensitivitySpecificityLoss(nn.Module):
    def __init__(self, alpha=0.5, eps=1e-6):
        super(SensitivitySpecificityLoss, self).__init__()
        self.alpha = alpha
        self.eps = eps

    def forward(self, logits, targets):
        targets = targets.float()
        probs = torch.sigmoid(logits)
        tp = (probs * targets).sum()
        fn = ((1 - probs) * targets).sum()
        sensitivity = tp / (tp + fn + self.eps)
        tn = ((1 - probs) * (1 - targets)).sum()
        fp = (probs * (1 - targets)).sum()
        specificity = tn / (tn + fp + self.eps)
        loss = 1 - (self.alpha * sensitivity + (1 - self.alpha) * specificity)
        return loss
    
class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, gamma=0.5, pos_weight=None, eps=1e-6):
        super(CombinedLoss, self).__init__()
        self.sens_spec_loss = SensitivitySpecificityLoss(alpha, eps)
        if pos_weight is not None:
            self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            self.bce_loss = nn.BCEWithLogitsLoss()
        self.beta = beta 
        self.gamma = gamma 

    def forward(self, logits, targets):
        bce_loss = self.bce_loss(logits, targets)
        sens_spec_loss = self.sens_spec_loss(logits, targets)
        combined_loss = self.beta * bce_loss + self.gamma * sens_spec_loss
        return combined_loss

num_pos = (y_train == 1).sum()
num_neg = (y_train == 0).sum()
pos_weight = num_neg / num_pos
pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32).to(device)

#Beta and gamma do not need to sum to 1, they are simply used to adjust the relative contribution of each component during training.
#You can try adjusting the parameters to see how they affect the results.
criterion = CombinedLoss(alpha=0.55, pos_weight=pos_weight_tensor)
train(model, train_loader, val_loader, device, optimizer, criterion)
test_accuracy = test(model, test_loader, device)

    
