# for every barrier file, get the max - min over barrier values
import argparse
from pathlib import Path
import torch
import pandas as pd
import numpy as np


## Setup
parser = argparse.ArgumentParser()
parser.add_argument('--dir', required=True, type=Path)
parser.add_argument('--pattern', default="barrier-*.pt", type=str)
parser.add_argument('--recursive', default=False, action="store_true")
parser.add_argument('--save_file', default=None, type=Path)
args = parser.parse_args()

stats = []

files = args.dir.rglob(args.pattern) if args.recursive else args.dir.glob(args.pattern)
for file in files:
    barriers = torch.load(file, map_location="cpu")
    for type, category in barriers.items():
        stats.append({
            "path": file,
            "type": type,
            **{k: max(v) if "loss" in k else min(v) if "acc" in k else len(v) for k, v in category.items()},
            **{k+".idx": np.argmax(v) if "loss" in k else np.argmin(v) if "acc" in k else len(v) for k, v in category.items()},
        })

# save stats together in one csv file
csv_file = args.dir / f"barriers.csv"  if args.save_file is None else args.save_file
df = pd.DataFrame(stats)
df.to_csv(csv_file)
print(f"Stats saved to {csv_file}")
