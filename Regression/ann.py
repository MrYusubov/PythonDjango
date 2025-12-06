import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv("transactions.csv")

X = df[["amount", "time_of_day", "location_distance", "is_international", "past_fraud_count"]].values
y = df["target"].values.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

class FraudDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_ds = FraudDataset(X_train, y_train)
test_ds = FraudDataset(X_test, y_test)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

class FraudNet(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        return self.net(x)

model = FraudNet(in_features=5).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = 50
for epoch in range(epochs):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        correct += (preds == yb).sum().item()
        total += yb.numel()

accuracy = 100 * correct / total
print(f"Accuracy: {accuracy:.2f}%")

new_tx = {
    "amount": 1200,
    "time_of_day": 2,
    "location_distance": 12.5,
    "is_international": 1,
    "past_fraud_count": 1
}

new_X = pd.DataFrame([new_tx])[["amount", "time_of_day", "location_distance", "is_international", "past_fraud_count"]].values
new_X = scaler.transform(new_X)
new_X_t = torch.tensor(new_X, dtype=torch.float32).to(device)

with torch.no_grad():
    risk_prob = torch.sigmoid(model(new_X_t)).item()

print(f"Risk faizi: {risk_prob*100:.2f}%")
