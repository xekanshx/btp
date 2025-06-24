import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --------------------- CONFIG ---------------------
SEQ_LEN = 7
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-3

# --------------------- LOAD + SCALE ---------------------
df = pd.read_csv("/Users/mac/btp/preprocessed_with_power.csv")
df_model = df.drop(columns=['Date'])

scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_model), columns=df_model.columns)

# --------------------- SEQUENCE CREATION ---------------------
def create_sequences(data, target_column, seq_len):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data.iloc[i - seq_len:i].values)
        y.append(data.iloc[i][target_column])
    return np.array(X), np.array(y)

X, y = create_sequences(df_scaled, target_column='Power', seq_len=SEQ_LEN)

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, shuffle=False)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False)

# --------------------- DATASET WRAPPER ---------------------
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_ds = TimeSeriesDataset(X_train, y_train)
val_ds = TimeSeriesDataset(X_val, y_val)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
test_ds = TimeSeriesDataset(X_test, y_test)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)


# --------------------- MODEL ---------------------
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, seq_len, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.positional_encoding = nn.Parameter(torch.randn(seq_len, d_model))
        self.input_projection = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.regressor = nn.Sequential(
            nn.Linear(seq_len * d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.input_projection(x) + self.positional_encoding
        x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2).contiguous().view(x.size(1), -1)
        return self.regressor(x).squeeze()

# --------------------- TRAINING FUNCTION ---------------------
def train_model(model, train_loader, val_loader, epochs, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            preds = model(xb)
            loss = loss_fn(preds, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                preds = model(xb)
                val_preds.extend(preds.numpy())
                val_true.extend(yb.numpy())

        mae = mean_absolute_error(val_true, val_preds)
        rmse = mean_squared_error(val_true, val_preds, squared=False)
        r2 = r2_score(val_true, val_preds)
        print(f"Epoch {epoch+1}: MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.3f}")

# --------------------- RUN ---------------------
model = TimeSeriesTransformer(
    input_dim=X_train.shape[2],
    seq_len=X_train.shape[1],
    d_model=64,
    nhead=4,
    num_layers=2,
    dropout=0.1
)

train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LR)
# --------------------- EVALUATE ON TEST SET ---------------------
model.eval()
test_preds, test_true = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        preds = model(xb)
        test_preds.extend(preds.numpy())
        test_true.extend(yb.numpy())

mae_test = mean_absolute_error(test_true, test_preds)
rmse_test = mean_squared_error(test_true, test_preds, squared=False)
r2_test = r2_score(test_true, test_preds)

print("\n----- Test Set Performance -----")
print(f"MAE:  {mae_test:.2f}")
print(f"RMSE: {rmse_test:.2f}")
print(f"R²:   {r2_test:.3f}")

# Optional: plot predictions vs actual
plt.figure(figsize=(10, 5))
plt.plot(test_true, label='Actual', color='blue')
plt.plot(test_preds, label='Predicted', color='red')
plt.title("Test Set: Actual vs Predicted Power Output")
plt.xlabel("Samples")
plt.ylabel("Power")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

