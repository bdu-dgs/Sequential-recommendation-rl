"""Generate (or verify) the simulated offline dataset for a config and seed, before any training job uses it.

Usage: python scripts/generate_data.py --config configs/experiment/base.yaml --seed 0 [key=value ...]
"""

import argparse
import json

import numpy as np

from seqrec_rl.training.runs import limit_threads, load_config, set_seed
from seqrec_rl.training.setup import OFFLINE_DIR, Setup

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment/base.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    limit_threads(4)
    set_seed(args.seed)
    S = Setup(cfg, args.seed)
    buf = S.offline_buffer()
    n = buf.size
    acts = buf.data["action"][:n]
    counts = np.bincount(acts, minlength=S.data.n_items)
    stats = {"dataset": S.dataset_name(), "transitions": int(n), "episodes": int(buf.data["done"][:n].sum()),
             "items_logged": int((counts > 0).sum()), "items_logged_10plus": int((counts >= 10).sum()),
             "logging": cfg["logging"], "sim": cfg["sim"]}
    (OFFLINE_DIR / f"{S.dataset_name()}.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats))
