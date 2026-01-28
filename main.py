"""HyperPep Eval-only entrypoint.

Loads a saved checkpoint (.pt) and evaluates it on a provided CSV.
This script is intentionally deterministic (as much as PyTorch allows) to make
evaluation reproducible across runs and machines.

Notes:
- The checkpoint is expected to contain a dict with at least 'model_state'.
- If the checkpoint also contains a 'config' dict, we prefer it over CLI args.
- Output metrics/curves are saved as a JSON for downstream plotting.

"""

# main.py — Eval-only: load a saved checkpoint and evaluate on a CSV
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"   # deterministic matmul

import argparse
import random
import numpy as np
import json
import re
import torch

# ====== Strict determinism ======
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from torch_geometric.loader import DataLoader

from device import device_info
from data import GeoDataset_1
from model import Hyperpep
from process import validation

# ---------------- CLI args ----------------
parser = argparse.ArgumentParser()
parser.add_argument("--test_name", type=str, required=True,
                    help="Eval CSV filename under ./data/ (e.g., NewTest.csv)")

# Load checkpoint (required)
parser.add_argument("--load_ckpt", type=str, required=True,
                    help="Path to a saved checkpoint .pt to load for evaluation.")

# Optional fallbacks (used only if checkpoint lacks config fields)
parser.add_argument("--mode", choices=["fg", "pg"], default="pg",
                    help="Feature extraction mode fallback if ckpt config missing.")
parser.add_argument("--k", type=int, default=3, help="k for pg mode fallback if ckpt config missing.")
parser.add_argument("--index_x", type=str, default="0,1,2",
                    help="Comma-separated CSV column indices used as model input. Fallback if ckpt config missing.")
parser.add_argument("--index_y", type=int, default=3,
                    help="CSV label column index (0-based). Fallback if ckpt config missing.")

# Output
parser.add_argument("--out_dir", type=str, default="metrics", help="Directory to save eval results JSON.")
parser.add_argument("--out_name", type=str, default="",
                    help="Optional output filename (e.g., eval_newtest.json). Default: auto from ckpt+test.")

args, _unknown = parser.parse_known_args()
# parse_known_args() lets you pass extra arguments (e.g., from notebooks)
# without hard-failing. Unrecognized args are stored in _unknown.


def parse_index_list(s: str):
    # Parse a comma-separated list like "0,1,2" -> [0, 1, 2].
    # Used to select which CSV columns are concatenated as input.

    s = str(s).strip()
    if not s:
        return []
    return [int(x) for x in s.split(",") if str(x).strip() != ""]

# ---------------- Seeds ----------------
GLOBAL_BASE_SEED = 42
def set_seed(seed: int = GLOBAL_BASE_SEED):
    # Set RNG seeds for Python / NumPy / PyTorch.
    # Keeping seeds fixed helps make preprocessing + evaluation reproducible.

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)

set_seed(GLOBAL_BASE_SEED)

# ---------------- Device ----------------
device_information = device_info()
# device_info() prints a short banner and exposes the torch.device.

print(device_information)
device = device_information.device

# ---------------- Data paths ----------------
DATA_DIR = "./data"
# By default we assume the evaluation CSV lives under ./data.
# (Change DATA_DIR if your data are stored elsewhere.)
eval_csv = os.path.join(DATA_DIR, args.test_name)

# ---------------- Checkpoint helpers ----------------
def load_checkpoint(path):
    # Checkpoint format contract:
    #   ckpt = {'model_state': state_dict, 'config': {...}, ...}

    ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict) or "model_state" not in ckpt:
        raise RuntimeError(f"Invalid checkpoint file: {path}")
    return ckpt

def initialize_model(dataset, hidden: int, drop: float, seed: int):
    # Build a Hyperpep model with input dims inferred from a dataset sample.
    # This keeps the model definition decoupled from feature extraction.

    torch.manual_seed(seed)
    sample = dataset[0]
    in_h   = sample.x_h.size(1)
    edge_h = sample.h2_edge_attr.size(1)
    model = Hyperpep(in_channels_h=in_h, edge_in_channels=edge_h, hidden=hidden, drop=drop).to(device)
    return model, dict(in_h=int(in_h), edge_h=int(edge_h))

def safe_stem(p: str):
    # Make a filesystem-friendly stem used to build the output JSON filename.

    base = os.path.basename(p)
    base = re.sub(r"\.pt$", "", base)
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", base)

# ---------------- Eval-only ----------------
def main():
    ckpt = load_checkpoint(args.load_ckpt)
    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}

    # Prefer config from checkpoint; fallback to CLI if missing
    feature_mode = str(cfg.get("feature_mode", args.mode)).lower().strip()
    pg_k = int(cfg.get("pg_k", args.k))
    idx_x = cfg.get("index_x", parse_index_list(args.index_x))
    idx_y = int(cfg.get("index_y", args.index_y))

    hidden = int(cfg.get("hidden", 160))
    drop   = float(cfg.get("dropout", 0.3))
    dims_ckpt = cfg.get("model_dims", None)
    bs = int(cfg.get("batch_size", 64 if feature_mode == "pg" else 128))

    print(f"[HyperPep][EVAL] load_ckpt={args.load_ckpt}")
    print(f"[HyperPep][EVAL] mode={feature_mode} k={pg_k} hidden={hidden} drop={drop} bs={bs}")
    print(f"[HyperPep][EVAL] index_x={idx_x}, index_y={idx_y}")

    # ------------------------------------------------------------
    # 1) Read config from checkpoint (preferred) / CLI (fallback)
    # ------------------------------------------------------------
    # Build eval dataset/loader
    eval_ds = GeoDataset_1(
        raw_name=eval_csv, root="",
        index_x=idx_x, index_y=idx_y, has_targets=True,
        hyper_mode=feature_mode, pg_k=pg_k
    )
    eval_loader = DataLoader(eval_ds, batch_size=bs, shuffle=False, num_workers=0)
    # shuffle=False: keep evaluation order stable.
    # num_workers=0: avoids multiprocessing nondeterminism / env issues.

    model, dims_built = initialize_model(eval_ds, hidden=hidden, drop=drop, seed=GLOBAL_BASE_SEED)

    # ------------------------------------------------------------
    # 2) Construct model and verify feature dimensions
    # ------------------------------------------------------------
    # Sanity-check feature dims (very helpful when mode/k mismatch)
    if isinstance(dims_ckpt, dict):
        in_h_ckpt = int(dims_ckpt.get("in_h", dims_built["in_h"]))
        edge_h_ckpt = int(dims_ckpt.get("edge_h", dims_built["edge_h"]))
        if in_h_ckpt != dims_built["in_h"] or edge_h_ckpt != dims_built["edge_h"]:
            raise RuntimeError(
                f"Checkpoint/model dims mismatch. ckpt={dims_ckpt} built={dims_built}. "
                f"Make sure eval uses the same feature mode/k and preprocessing as training."
            )

    # Load trained weights (strict=True to catch missing/unexpected keys).
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device)
    model.eval()

    # Run evaluation.
    # We pass epoch=0 only because the validation() helper prints an epoch field;
    # this script is eval-only and does not iterate epochs.
    loss, AUC, AUPR, fpr, tpr, precision, recall, y_true, y_prob = validation(model, device, eval_loader, epoch=0)

    # Collect both scalar metrics and full curves/probabilities so you can
    # re-plot ROC/PR later without re-running the model.
    out = {
        "eval_csv": eval_csv,
        "checkpoint": args.load_ckpt,
        "mode": feature_mode,
        "pg_k": pg_k,
        "index_x": idx_x,
        "index_y": idx_y,
        "hidden": hidden,
        "dropout": drop,
        "batch_size": bs,
        "loss": float(loss),
        "AUC": float(AUC) if AUC == AUC else None,   # handle NaN (NaN != NaN)

        "AUPR": float(AUPR) if AUPR == AUPR else None,
        "fpr": [float(x) for x in (fpr if fpr is not None else [])],
        "tpr": [float(x) for x in (tpr if tpr is not None else [])],
        "precision": [float(x) for x in (precision if precision is not None else [])],
        "recall": [float(x) for x in (recall if recall is not None else [])],
        "y_true": [float(x) for x in (y_true if y_true is not None else [])],
        "y_prob": [float(x) for x in (y_prob if y_prob is not None else [])],
        "deterministic": True,
        "global_seed": GLOBAL_BASE_SEED,
    }

    os.makedirs(args.out_dir, exist_ok=True)
    # ------------------------------------------------------------
    # 3) Write results to disk (JSON)
    # ------------------------------------------------------------
    if args.out_name.strip():
        out_path = os.path.join(args.out_dir, args.out_name.strip())
    else:
        out_path = os.path.join(args.out_dir, f"eval_{safe_stem(args.load_ckpt)}__{safe_stem(args.test_name)}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
