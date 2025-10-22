"""
model_baseline_B_multilingual.py

- Loads preprocessed spectrogram .npy produced by preprocess_hdtrack2_multilingual.py
- Treats each language+category label as a distinct class (label strings like 'en__Follow-up Questions')
- LSTM classifier (biLSTM optional), training loop, evaluation, saves plots and best model.

Usage:
 python model_baseline_B_multilingual.py --data_dir preprocessed --epochs 25 --batch_size 32
"""
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import csv
from tqdm import tqdm
import matplotlib.pyplot as plt

def ensure_dir(p): os.makedirs(p, exist_ok=True)

class SpectrogramDataset(Dataset):
    def __init__(self, manifest_csv, label_encoder=None, max_time_steps=400):
        self.items = []
        with open(manifest_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for r in reader:
                self.items.append((r['filepath'], r['label']))
        self.labels = [lab for _, lab in self.items]
        self.le = label_encoder if label_encoder is not None else LabelEncoder().fit(self.labels)
        self.max_time_steps = max_time_steps

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        spec = np.load(path)  # (freq_bins, time_steps)
        spec_t = spec.T  # (time_steps, freq_bins)
        t_len, freq_bins = spec_t.shape
        if t_len >= self.max_time_steps:
            spec_t = spec_t[:self.max_time_steps, :]
        else:
            pad = np.zeros((self.max_time_steps - t_len, freq_bins), dtype=spec_t.dtype)
            spec_t = np.vstack([spec_t, pad])
        x = torch.from_numpy(spec_t).float()
        y = int(self.le.transform([label])[0])
        return x, y

def collate_batch(batch):
    xs = torch.stack([b[0] for b in batch], dim=0)
    ys = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return xs, ys

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, num_classes, bidirectional=False, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True,
                            bidirectional=bidirectional, dropout=dropout if num_layers>1 else 0.0)
        self.fc = nn.Linear(hidden_size * (2 if bidirectional else 1), num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)  # (batch, time, hidden*dirs)
        last = out[:, -1, :]
        logits = self.fc(last)
        return logits

def train_one_epoch(model, loader, opt, criterion, device):
    model.train()
    total_loss = 0.0
    preds, targets = [], []
    for x, y in tqdm(loader, desc="Train"):
        x = x.to(device); y = y.to(device)
        opt.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        opt.step()
        total_loss += loss.item() * x.size(0)
        preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
        targets.extend(y.cpu().tolist())
    return total_loss / len(loader.dataset), accuracy_score(targets, preds)

def eval_model(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    preds, targets = [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Eval"):
            x = x.to(device); y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
            targets.extend(y.cpu().tolist())
    avg_loss = total_loss / len(loader.dataset) if len(loader.dataset)>0 else 0.0
    acc = accuracy_score(targets, preds) if len(targets)>0 else 0.0
    f1 = f1_score(targets, preds, average='macro') if len(set(targets))>1 else 0.0
    return avg_loss, acc, f1, targets, preds

def save_plots(history, out_dir):
    ensure_dir(out_dir)
    plt.figure(); plt.plot(history['train_loss'], label='train_loss'); plt.plot(history['val_loss'], label='val_loss'); plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(out_dir,'loss.pdf')); plt.close()
    plt.figure(); plt.plot(history['train_acc'], label='train_acc'); plt.plot(history['val_acc'], label='val_acc'); plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(out_dir,'accuracy.pdf')); plt.close()

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print("Device:", device)

    train_manifest = os.path.join(args.data_dir, "train_manifest.csv")
    test_manifest = os.path.join(args.data_dir, "test_manifest.csv")
    if not os.path.isfile(train_manifest) or not os.path.isfile(test_manifest):
        raise RuntimeError("Run preprocessing first; manifests missing.")

    # build label encoder from train manifest
    # build label encoder from train + test labels
    labels = []

    # read train labels
    with open(train_manifest, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            labels.append(r['label'])

    # read test labels also
    with open(test_manifest, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            labels.append(r['label'])

    if not labels:
        raise RuntimeError("No labels found.")

    # fit once on ALL labels
    le = LabelEncoder().fit(labels)
    num_classes = len(le.classes_)
    print("Found classes (count):", num_classes)
    print("Classes:", le.classes_)


    train_ds = SpectrogramDataset(train_manifest, label_encoder=le, max_time_steps=args.max_time_steps)
    val_ds = SpectrogramDataset(test_manifest, label_encoder=le, max_time_steps=args.max_time_steps)

    sample_x, _ = train_ds[0]
    input_dim = sample_x.shape[1]
    print("Input freq bins:", input_dim)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch, num_workers=0)

    model = LSTMClassifier(input_dim=input_dim,
                           hidden_size=args.hidden_size,
                           num_layers=args.num_layers,
                           num_classes=num_classes,
                           bidirectional=args.bidirectional,
                           dropout=args.dropout).to(device)
    print(model)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0.0
    ensure_dir("models")
    plots_dir = os.path.join("plots", "lstm_spectrogram")
    ensure_dir(plots_dir)

    for epoch in range(1, args.epochs+1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, opt, criterion, device)
        val_loss, val_acc, val_f1, _, _ = eval_model(model, val_loader, criterion, device)
        print(f"Train loss {train_loss:.4f} acc {train_acc:.4f}")
        print(f"Val   loss {val_loss:.4f} acc {val_acc:.4f} f1 {val_f1:.4f}")
        history['train_loss'].append(train_loss); history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc); history['val_acc'].append(val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join("models", "lstm_multilingual_labelled.pt"))
            print("Saved best model.")

    # final eval
    val_loss, val_acc, val_f1, targets, preds = eval_model(model, val_loader, criterion, device)
    print("\nFinal test eval:")
    print(f"Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
    print(classification_report(targets, preds, target_names=le.classes_))

    cm = confusion_matrix(targets, preds)
    with open(os.path.join(plots_dir, "eval_metrics.txt"), 'w', encoding='utf-8') as f:
        f.write(f"Accuracy: {val_acc:.6f}\nF1: {val_f1:.6f}\n\n")
        f.write(classification_report(targets, preds, target_names=le.classes_))
        f.write("\nConfusion Matrix:\n")
        f.write(np.array2string(cm))
    save_plots(history, plots_dir)
    print(f"Saved plots & metrics to {plots_dir}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="preprocessed", help="Directory with train_manifest.csv/test_manifest.csv")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--hidden_size", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--bidirectional", action='store_true')
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--max_time_steps", type=int, default=400)
    p.add_argument("--no_cuda", action='store_true')
    args = p.parse_args()
    main(args)