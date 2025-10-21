import argparse, os, json, pickle, re, html, urllib.parse
import numpy as np
import torch
import torch.nn as nn

# ---------- CLI ----------
p = argparse.ArgumentParser(description="Predict XSS probability for new payloads/URLs.")
p.add_argument("--out_dir", default="./out", help="Folder containing mlp_state.pth, vocab.pkl, meta.json")
p.add_argument("--text", nargs="*", help="One or more payloads/URLs to score")
p.add_argument("--file", help="Path to a text file with one payload per line (optional)")
p.add_argument("--threshold", type=float, default=None, help="Override decision threshold (default: meta.json best_threshold or 0.5)")
args = p.parse_args()

out_dir = os.path.abspath(args.out_dir)

# ---------- Load artifacts ----------
meta_path = os.path.join(out_dir, "meta.json")
vocab_path = os.path.join(out_dir, "vocab.pkl")
state_path = os.path.join(out_dir, "mlp_state.pth")

if not (os.path.exists(meta_path) and os.path.exists(vocab_path) and os.path.exists(state_path)):
    raise SystemExit(f"Missing artifacts in {out_dir}. Expected: mlp_state.pth, vocab.pkl, meta.json")

with open(meta_path, "r", encoding="utf-8") as f:
    meta = json.load(f)
with open(vocab_path, "rb") as f:
    voc = pickle.load(f)

in_dim = int(meta.get("in_dim", 5))
hid = int(meta.get("hid", 64))
best_thr = float(meta.get("best_threshold", 0.5))
tok2id = voc.get("tok2id", {})
bow_dim = len(tok2id)

# ---------- Preprocessing (match training) ----------
def decode(s: str) -> str:
    return html.unescape(urllib.parse.unquote_plus(str(s)))

strong = re.compile(
    r'(<script|%3cscript|&lt;script|\bonon[a-z]+\s*=|onerror%3d|onload%3d|javascript:|javascript%3a|alert\()',
    re.I
)

def req_stat_features(s: str):
    s = str(s).lower()
    L = len(s)
    return np.array([
        np.log1p(L),
        s.count('%') / max(L, 1),
        (s.count('<') + s.count('>')) / max(L, 1),
        (s.count('"') + s.count("'")) / max(L, 1),
        1.0 if strong.search(s) else 0.0
    ], dtype=np.float32)

split_re = re.compile(r'[^a-zA-Z0-9]+')
def bow_vectorize(s: str, tok2id: dict):
    if not tok2id:
        return np.zeros(0, dtype=np.float32)
    vec = np.zeros(len(tok2id), dtype=np.float32)
    seen = set()
    for t in (t for t in split_re.split(str(s).lower()) if len(t) >= 3):
        j = tok2id.get(t)
        if j is not None and j not in seen:
            vec[j] = 1.0
            seen.add(j)
    return vec

def vectorize(texts):
    Xs = []
    for t in texts:
        raw = decode(t)
        stat = req_stat_features(raw)
        bow = bow_vectorize(raw, tok2id)
        x = np.hstack([stat, bow]).astype(np.float32) if bow_dim else stat
        Xs.append(x)
    X = np.vstack(Xs).astype(np.float32)
    # pad/trim to match expected in_dim (in case vocab differs)
    if X.shape[1] != in_dim:
        if X.shape[1] < in_dim:
            pad = np.zeros((X.shape[0], in_dim - X.shape[1]), dtype=np.float32)
            X = np.hstack([X, pad])
        else:
            X = X[:, :in_dim]
    return X

# ---------- Model (match training) ----------
class MLP(nn.Module):
    def __init__(self, in_dim: int, hid: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid), nn.ReLU(),
            nn.Linear(hid, hid//2), nn.ReLU(),
            nn.Linear(hid//2, 1)
        )
    def forward(self, x): return self.net(x).squeeze(-1)

model = MLP(in_dim=in_dim, hid=hid)
# robust load across torch versions
state = torch.load(state_path, map_location="cpu")
if isinstance(state, dict) and "state_dict" in state:
    state = state["state_dict"]
model.load_state_dict(state, strict=False)
model.eval()

# ---------- Collect inputs ----------
texts = []
if args.text:
    texts.extend(args.text)
if args.file:
    with open(args.file, "r", encoding="utf-8", errors="ignore") as f:
        texts.extend([line.strip() for line in f if line.strip()])

if not texts:
    raise SystemExit("No input provided. Use --text 'payload here' or --file requests.txt")

# ---------- Predict ----------
X = torch.tensor(vectorize(texts))
with torch.no_grad():
    probs = torch.sigmoid(model(X)).numpy()

thr = float(args.threshold) if args.threshold is not None else best_thr
preds = (probs >= thr).astype(int)

# ---------- Print (minimal) ----------
# prob, pred(0/1), text
for t, p, y in zip(texts, probs, preds):
    # trim long texts for console readability
    short = (t[:140] + "...") if len(t) > 140 else t
    print(f"{p:.3f}\t{y}\t{short}")
