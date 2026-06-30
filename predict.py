import os
import torch
import torch.nn as nn
import pickle

from preprocessing import clean_text

# -----------------------------
# Load Vocabulary
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VOCAB_PATH = os.path.join(BASE_DIR, "vocab.pkl")
MODEL_PATH = os.path.join(BASE_DIR, "cyberbullying_model.pth")

with open(VOCAB_PATH, "rb") as f:
    vocab = pickle.load(f)

MAX_LEN = 30


def encode(sentence):
    sentence = clean_text(sentence)
    tokens = sentence.split()

    ids = [vocab.get(word, vocab["<UNK>"]) for word in tokens]

    if len(ids) < MAX_LEN:
        ids += [0] * (MAX_LEN - len(ids))
    else:
        ids = ids[:MAX_LEN]

    return torch.tensor([ids], dtype=torch.long)


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


# -----------------------------
# Load Model
# -----------------------------
model = LSTMClassifier()

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=torch.device("cpu"))
)
model.eval()


# -----------------------------
# Prediction Function
# -----------------------------
def predict(text):

    encoded = encode(text)

    with torch.no_grad():

        output = model(encoded).item()

    if output >= 0.5:
        return "Cyberbullying", output
    else:
        return "Non-Cyberbullying", output


# -----------------------------
# Test
# -----------------------------
if __name__ == "__main__":

    while True:

        sentence = input("\nEnter Comment (type 'exit' to quit): ")

        if sentence.lower() == "exit":
            break

        result, confidence = predict(sentence)

        print(f"\nPrediction : {result}")
        print(f"Confidence : {confidence:.4f}")