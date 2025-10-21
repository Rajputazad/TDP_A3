#!/usr/bin/env python3


import os, re, html, argparse, urllib.parse, json, pickle, logging
from typing import List, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support
from tqdm import trange

# ---------------- argparse ----------------
parser = argparse.ArgumentParser(description="XSS detection without DGL (PyTorch MLP + optional BoW).")
parser.add_argument("--csv", "-c", required=True, help="Path to CSV file")
parser.add_argument("--payload_col", "-p", default=None, help="Column name with request/payload (auto-detect if omitted)")
parser.add_argument("--vocab_size", type=int, default=2000, help="Top tokens to include in BoW vocab (0 to disable BoW)")
parser.add_argument("--seed", type=int, default=7, help="Random seed")
parser.add_argument("--out_dir", type=str, default="./out", help="Output directory for artifacts")
parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
parser.add_argument("--hid", type=int, default=64, help="Hidden dimension for MLP")
parser.add_argument("--topk", type=int, default=20, help="How many suspicious samples to save")
args = parser.parse_args()

CSV_PATH = os.path.abspath(args.csv)
PAYLOAD_COL = args.payload_col
VOCAB_SIZE = max(0, int(args.vocab_size))
SEED = args.seed
OUT_DIR = os.path.abspath(args.out_dir)
EPOCHS = args.epochs
LR = args.lr
HID = args.hid
TOPK = args.topk
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------- logging ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("xss_nodgl")

# ---------------- helpers ----------------
def decode(s: str) -> str:
    if pd.isna(s):
        return ""
    return html.unescape(urllib.parse.unquote_plus(str(s)))

def load_csv(p: str) -> pd.DataFrame:
    if not os.path.exists(p):
        raise FileNotFoundError(f"CSV not found: {p}")
    try:
        return pd.read_csv(p)
    except UnicodeDecodeError:
        log.warning("UnicodeDecodeError reading CSV; trying latin-1 encoding.")
        return pd.read_csv(p, encoding="latin-1")

def build_payload(df: pd.DataFrame, col: str = None) -> pd.Series:
    lowers = {c.lower(): c for c in df.columns}
    if col:
        key = lowers.get(col.lower(), None)
        if key is None:
            raise ValueError(f"Column '{col}' not found in CSV: {list(df.columns)}")
        return df[key].astype(str).apply(decode)
    pref = ['payload','request','url','params','query','data','body','path','full_request']
    found = [lowers[p] for p in pref if p in lowers]
    if found:
        return df.apply(lambda r: decode(" ".join(str(r[c]) for c in found if pd.notnull(r[c]))), axis=1)
    return df[df.columns[0]].astype(str).apply(decode)

# basic tokenization for BoW
split_re = re.compile(r'[^a-zA-Z0-9]+')

# Regex for weak labels (and a feature)
strong = re.compile(
    r'(<script|%3cscript|&lt;script|\bonon[a-z]+\s*=|onerror%3d|onload%3d|javascript:|javascript%3a|alert\()',
    re.I
)

def req_stat_features(s: str) -> List[float]:
    s = str(s).lower()
    L = len(s)
    return [
        np.log1p(L),
        s.count('%') / max(L, 1),
        (s.count('<') + s.count('>')) / max(L, 1),
        (s.count('"') + s.count("'")) / max(L, 1),
        1.0 if strong.search(s) else 0.0
    ]

def build_vocab(payload: pd.Series, k: int) -> Tuple[List[str], dict]:
    if k <= 0:
        return [], {}
    freq = {}
    for s in payload:
        for t in (t for t in split_re.split(str(s).lower()) if len(t) >= 3):
            freq[t] = freq.get(t, 0) + 1
    vocab = [t for t,_ in sorted(freq.items(), key=lambda x: -x[1])[:k]]
    tok2id = {t:i for i,t in enumerate(vocab)}
    return vocab, tok2id

def bow_vectorize(s: str, tok2id: dict) -> np.ndarray:
    vec = np.zeros(len(tok2id), dtype=np.float32)
    if not tok2id:
        return vec
    seen = set()
    for t in (t for t in split_re.split(str(s).lower()) if len(t) >= 3):
        j = tok2id.get(t)
        if j is not None and j not in seen:
            vec[j] = 1.0
            seen.add(j)
    return vec

# ---------------- model ----------------
class MLP(nn.Module):
    def __init__(self, in_dim: int, hid: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid), nn.ReLU(),
            nn.Linear(hid, hid//2), nn.ReLU(),
            nn.Linear(hid//2, 1)
        )
    def forward(self, x):  # x: (B, D)
        return self.net(x).squeeze(-1)

def metrics_from_logits(logits: torch.Tensor, y_true: torch.Tensor, thr: float = 0.5):
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    preds = (probs >= thr).astype(int)
    yt = y_true.detach().cpu().numpy()
    p, r, f1, _ = precision_recall_fscore_support(yt, preds, average='binary', zero_division=0)
    return float(p), float(r), float(f1)

def tune_threshold(logits: torch.Tensor, y_true: torch.Tensor, grid=None):
    if grid is None:
        grid = np.linspace(0.2, 0.8, 25)  # wide enough sweep
    best_thr, best_f1 = 0.5, -1.0
    for t in grid:
        p, r, f1 = metrics_from_logits(logits, y_true, thr=float(t))
        if f1 > best_f1:
            best_f1, best_thr = f1, float(t)
    return best_thr, best_f1

# ---------------- main ----------------
def main():
    # Repro
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Load CSV & build payload
    log.info("Loading CSV: %s", CSV_PATH)
    raw = load_csv(CSV_PATH)
    payload = build_payload(raw, PAYLOAD_COL)
    if len(payload) == 0:
        raise RuntimeError("No payload rows found after loading CSV.")

    # Weak labels
    labels = payload.str.lower().str.contains(strong, na=False).astype(int).values
    pos = int(labels.sum()); tot = len(labels)
    log.info("Weak labels: %d positives out of %d (%.2f%%)", pos, tot, 100.0*pos/max(1,tot))

    # Build vocab (optional)
    vocab, tok2id = build_vocab(payload, VOCAB_SIZE)
    bow_dim = len(vocab)
    log.info("BoW vocab size = %d", bow_dim)

    # Features
    stat_feats = np.vstack([req_stat_features(s) for s in payload]).astype(np.float32)
    if bow_dim > 0:
        bow = np.vstack([bow_vectorize(s, tok2id) for s in payload]).astype(np.float32)
        X = np.hstack([stat_feats, bow]).astype(np.float32)
    else:
        X = stat_feats

    y = torch.tensor(labels, dtype=torch.float32)

    # Train/val/test split
    n = len(payload)
    idx = np.arange(n); np.random.shuffle(idx)
    n_val = int(0.15*n); n_test = int(0.15*n)
    val_idx = idx[:n_val]; test_idx = idx[n_val:n_val+n_test]; train_idx = idx[n_val+n_test:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train = torch.tensor(X_train, device=device)
    X_val   = torch.tensor(X_val,   device=device)
    X_test  = torch.tensor(X_test,  device=device)
    y_train = y_train.to(device)
    y_val   = y_val.to(device)
    y_test  = y_test.to(device)

    # Model, loss, opt
    in_dim = X.shape[1]
    model = MLP(in_dim=in_dim, hid=HID).to(device)

    # Class imbalance handling
    pos_weight = None
    if y_train.sum().item() > 0 and (y_train.shape[0] - y_train.sum().item()) > 0:
        # pos_weight = (#neg / #pos)
        num_pos = y_train.sum().item()
        num_neg = y_train.shape[0] - num_pos
        pos_weight = torch.tensor([max(1.0, num_neg/num_pos)], device=device)
        log.info("Using pos_weight = %.3f", pos_weight.item())

    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    # Train
    log.info("Training for %d epochs on %s (in_dim=%d, bow_dim=%d)", EPOCHS, device, in_dim, bow_dim)
    for ep in trange(1, EPOCHS+1, desc="epochs"):
        model.train()
        logits = model(X_train)
        loss = bce(logits, y_train)
        opt.zero_grad(); loss.backward(); opt.step()

    # Eval on val to tune threshold
    model.eval()
    with torch.no_grad():
        val_logits = model(X_val)
        best_thr, best_f1 = tune_threshold(val_logits, y_val)
        te_logits = model(X_test)
        te_p, te_r, te_f1 = metrics_from_logits(te_logits, y_test, thr=best_thr)
    log.info("Best threshold on VAL = %.3f (F1=%.3f) | TEST P=%.3f R=%.3f F1=%.3f",
             best_thr, best_f1, te_p, te_r, te_f1)

    # Score all & save top suspicious
    with torch.no_grad():
        all_probs = torch.sigmoid(model(torch.tensor(X, device=device))).detach().cpu().numpy()
    top_idx = np.argsort(-all_probs)[:TOPK]
    sample_file = os.path.join(OUT_DIR, "top_suspicious.txt")
    with open(sample_file, "w", encoding="utf-8") as fh:
        fh.write("idx\tprob\tsample\n")
        for i in top_idx:
            line = str(payload.iloc[i]).replace("\n"," ")
            fh.write(f"{i}\t{all_probs[i]:.6f}\t{line[:400]}\n")
    log.info("Saved top suspicious requests to %s", sample_file)

    # Save model + artifacts
    model_path = os.path.join(OUT_DIR, "mlp_state.pth")
    torch.save(model.state_dict(), model_path)

    with open(os.path.join(OUT_DIR, "vocab.pkl"), "wb") as f:
        pickle.dump({'vocab': vocab, 'tok2id': tok2id}, f)

    meta = {
        "best_threshold": best_thr,
        "val_best_f1": best_f1,
        "test_precision": te_p,
        "test_recall": te_r,
        "test_f1": te_f1,
        "in_dim": in_dim,
        "bow_dim": bow_dim,
        "stat_dim": 5,
        "epochs": EPOCHS,
        "seed": SEED,
        "csv": CSV_PATH,
        "payload_col": PAYLOAD_COL,
    }
    with open(os.path.join(OUT_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    log.info("Saved model to %s, vocab.pkl and meta.json to %s", model_path, OUT_DIR)

if __name__ == "__main__":
    main()
