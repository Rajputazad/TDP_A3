"""
xss_detector.py

Train a DGL-free XSS detector using request-level features + optional bag-of-words
and a small PyTorch MLP. Auto-tunes decision threshold on the validation set.

Usage (train):
python xss_detector.py --csv ./dataset/csic_database.csv --out_dir ./out --payload_col URL

  # if needed: --payload_col URL
  # optional:  --vocab_size 2000 --epochs 50

Artifacts:
  out/
    top_suspicious.txt         # top 20 requests by predicted probability
    mlp_state.pth              # model state dict
    vocab.pkl                  # {'vocab': [...], 'tok2id': {...}}
    meta.json                  # meta info incl. best_threshold

Requirements:
  pip install torch pandas numpy scikit-learn tqdm
"""

# How to run
python xss_detector.py --csv ./dataset/csic_database.csv --out_dir ./out --payload_col URL

# For single script
python predict.py --out_dir ./out --text "http://site/?q=<script>alert(1)</script>"



#Multiple scripts
python predict.py --out_dir ./out --text "a=1&b=2" "javascript:alert(1)" "/search?q=%3Cscript%3Ealert(1)%3C%2Fscript%3E"