import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter
import pickle

from preprocessing import clean_text

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("../dataset/cyberbullying.csv")

# Select required columns
df = df[["tweet_text", "cyberbullying_type"]]

# Rename columns
df.columns = ["text", "label"]

# Convert multiclass labels into binary labels
# not_cyberbullying = 0
# all other classes = 1
df["label"] = df["label"].apply(
    lambda x: 0 if x == "not_cyberbullying" else 1
)

# Clean text
df["text"] = df["text"].astype(str).apply(clean_text)

texts = df["text"].tolist()
labels = df["label"].tolist()

print(df.head())
print(df["label"].value_counts())
print("Total Samples:", len(df))

# -----------------------------
# Build Vocabulary
# -----------------------------
counter = Counter()

for text in texts:
    counter.update(text.split())

vocab = {"<PAD>": 0, "<UNK>": 1}

for word, count in counter.items():
    if count >= 1:
        vocab[word] = len(vocab)

MAX_LEN = 30

def encode(sentence):
    tokens = sentence.split()

    ids = []

    for token in tokens:
        ids.append(vocab.get(token, vocab["<UNK>"]))

    if len(ids) < MAX_LEN:
        ids += [0] * (MAX_LEN - len(ids))
    else:
        ids = ids[:MAX_LEN]

    return ids

X = [encode(x) for x in texts]
y = labels

# Save Vocabulary
with open("vocab.pkl", "wb") as f:
    pickle.dump(vocab, f)

# -----------------------------
# Train Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
# -----------------------------
# Dataset Class
# -----------------------------
class CyberDataset(Dataset):

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_dataset = CyberDataset(X_train, y_train)
test_dataset = CyberDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

# -----------------------------
# LSTM Model
# -----------------------------
class LSTMClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.embedding = nn.Embedding(len(vocab), 128)

        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            batch_first=True
        )

        self.fc = nn.Linear(128, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x = self.embedding(x)

        output, (hidden, cell) = self.lstm(x)

        hidden = hidden[-1]

        out = self.fc(hidden)

        out = self.sigmoid(out)

        return out.squeeze()


model = LSTMClassifier()

criterion = nn.BCELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# -----------------------------
# Training
# -----------------------------
EPOCHS = 20

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    for inputs, labels in train_loader:

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(
        f"Epoch {epoch+1}/{EPOCHS} Loss = {total_loss:.4f}"
    )

# -----------------------------
# Evaluation
# -----------------------------
model.eval()

predictions = []
actual = []

with torch.no_grad():

    for inputs, labels in test_loader:

        outputs = model(inputs)

        preds = (outputs >= 0.5).int()

        predictions.extend(preds.tolist())

        actual.extend(labels.int().tolist())

accuracy = accuracy_score(actual, predictions)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

# -----------------------------
# Save Model
# -----------------------------
torch.save(model.state_dict(), "cyberbullying_model.pth")

print("\nModel Saved Successfully.")
print("Vocabulary Saved Successfully.")