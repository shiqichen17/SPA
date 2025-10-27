#!/usr/bin/env python3
"""
Filter SFT parquet splits by minimal structural rules:

Keep only rows whose response follows this structure (case-insensitive tags):
- Contains '<think> ... </think>'
- Inside think: at least one '<observation> ... </observation>' and at least one
  '<prediction> ... </prediction>', with the first observation before the first prediction

Optional length filter and de-duplication are still supported.

Usage:
python filter_sft_by_tags.py \
--in-train /projects/b1222/shiqi/Ragen-dev-test/ragen/sft/data/sudoku_coords_0920_1756/wm_train.parquet \
--in-val   /projects/b1222/shiqi/Ragen-dev-test/ragen/sft/data/sudoku_coords_0920_1756/wm_val.parquet \
--out-dir  /projects/b1222/shiqi/Ragen-dev-test/ragen/sft/data/sudoku_filtered \
--min-len  1 --max-len 100000 --no-dedup  # all optional
Requires: pandas, pyarrow
"""

import argparse
import os
import re
from typing import Optional, Tuple

import pandas as pd


def load_parquet(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")
    try:
        return pd.read_parquet(path)
    except Exception as e:
        raise RuntimeError(f"Failed to read parquet {path}: {e}")


_DIR_PATTERN = re.compile(r"^\s*(?:Up|Down|Left|Right)(?:\s*\|\|\s*(?:Up|Down|Left|Right)){0,19}\s*$",
                          flags=re.IGNORECASE)


def _extract_tag(text: str, tag: str) -> Tuple[Optional[re.Match], Optional[str]]:
    m = re.search(fr"<{tag}>(.*?)</{tag}>", text, flags=re.IGNORECASE | re.DOTALL)
    return m, (m.group(1) if m else None)


def is_good_response(resp: str) -> bool:
    """Apply basic structural checks to a response string."""
    if not isinstance(resp, str) or resp.strip() == "":
        return False

    # Must have complete structure: <think>...</think><answer>...</answer>
    # And think must appear before answer
    m_think, think_body = _extract_tag(resp, "think")
    m_answer, answer_body = _extract_tag(resp, "answer")
    
    if not m_think or not m_answer:
        return False
    
    # Check that think appears before answer
    if m_think.start() >= m_answer.start():
        return False

    # Inside think: need at least one observation and one prediction
    # And first observation must appear before first prediction
    if not think_body:
        return False
    
    obs_iter = list(re.finditer(r"<observation>(.*?)</observation>", think_body, flags=re.IGNORECASE | re.DOTALL))
    pred_iter = list(re.finditer(r"<prediction>(.*?)</prediction>", think_body, flags=re.IGNORECASE | re.DOTALL))
    
    # Require both observation and prediction, and observation must appear first
    if len(obs_iter) == 0 or len(pred_iter) == 0:
        return False
    if obs_iter[0].start() > pred_iter[0].start():
        return False

    return True


def filter_df(df: pd.DataFrame, min_len: int, max_len: int, dedup: bool) -> pd.DataFrame:
    # Basic column presence
    needed = ["prompt", "response"]
    for col in needed:
        if col not in df.columns:
            raise KeyError(f"Input dataframe missing required column: {col}")

    # Drop NA
    df = df.dropna(subset=["prompt", "response"]).copy()

    # Ensure strings
    df["response"] = df["response"].astype(str)

    # Strict structural filter
    mask_struct = df["response"].apply(is_good_response)
    df = df[mask_struct]

    # Length filter
    if min_len is not None or max_len is not None:
        min_len = 0 if min_len is None else min_len
        max_len = 10**12 if max_len is None else max_len
        lens = df["response"].str.len()
        df = df[(lens >= min_len) & (lens <= max_len)]

    # Dedup on (prompt, response)
    if dedup:
        df = df.drop_duplicates(subset=["prompt", "response"])

    return df.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-train", required=True, help="Input train parquet path")
    ap.add_argument("--in-val", required=True, help="Input val parquet path")
    ap.add_argument("--out-dir", required=True, help="Output directory for filtered parquet files")
    ap.add_argument("--min-len", type=int, default=None, help="Min response length filter (chars)")
    ap.add_argument("--max-len", type=int, default=None, help="Max response length filter (chars)")
    ap.add_argument("--no-dedup", action="store_true", help="Disable deduplication on (prompt, response)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading train: {args.in_train}")
    train = load_parquet(args.in_train)
    print(f"Loading val  : {args.in_val}")
    val = load_parquet(args.in_val)

    print(f"Train size before: {len(train)}; Val size before: {len(val)}")
    train_f = filter_df(train, args.min_len, args.max_len, dedup=(not args.no_dedup))
    val_f = filter_df(val, args.min_len, args.max_len, dedup=(not args.no_dedup))

    # # only keep 1/7 of the train and valid
    # train_f = train_f.sample(frac=1/7)
    # val_f = val_f.sample(frac=1/7)

    print(f"Train size after : {len(train_f)}; Val size after : {len(val_f)}")

    out_train = os.path.join(args.out_dir, "wm_train.parquet")
    out_val = os.path.join(args.out_dir, "wm_val.parquet")

    train_f.to_parquet(out_train, index=False)
    val_f.to_parquet(out_val, index=False)

    print(f"Saved filtered train: {out_train}")
    print(f"Saved filtered val  : {out_val}")


if __name__ == "__main__":
    main()