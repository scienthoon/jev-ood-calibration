"""
scripts/convert.py — build the JSONL used here from public HuggingFace datasets.

    python scripts/convert.py hf --preset openbookqa     --out data/ho_obqa --val-limit 2000
    python scripts/convert.py hf --preset commonsense_qa --out data/ho_csqa --val-limit 2000
    python scripts/convert.py hf --preset hellaswag      --out data/ho_hs   --val-limit 2000

Only <out>/val.jsonl was used in this study. Requires `pip install datasets`.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple


def write_jsonl(path: str, records: Sequence[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _load_hf(name: str, config: Optional[str], split: str):
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("pip install datasets 가 필요합니다")
    if config:
        return load_dataset(name, config, split=split)
    return load_dataset(name, split=split)


def preset_commonsense_qa(split: str, limit: Optional[int]) -> List[Dict[str, Any]]:
    """5지선다 상식 (MIT). held-out 용."""
    dataset = _load_hf("tau/commonsense_qa", None, split)
    records: List[Dict[str, Any]] = []
    for i, row in enumerate(dataset):
        if limit is not None and i >= limit:
            break
        labels = list(row["choices"]["label"])
        texts = list(row["choices"]["text"])
        options = {label: text for label, text in zip(labels, texts)}
        answer = str(row["answerKey"])
        if answer not in options:
            continue
        records.append({
            "state": row["question"],
            "type": "choice",
            "question": "Which option is the correct answer?",
            "options": options,
            "label": answer,
            "source": "commonsense_qa",
        })
    return records


def preset_hellaswag(split: str, limit: Optional[int]) -> List[Dict[str, Any]]:
    """4지선다, 선택지가 긴 문장 (MIT). held-out 용."""
    dataset = _load_hf("Rowan/hellaswag", None, split)
    records: List[Dict[str, Any]] = []
    for i, row in enumerate(dataset):
        if limit is not None and i >= limit:
            break
        label = str(row["label"]).strip()
        if not label.isdigit():
            continue
        endings = list(row["endings"])
        if not (0 <= int(label) < len(endings)):
            continue
        options = {str(k): ending for k, ending in enumerate(endings)}
        records.append({
            "state": row["ctx"],
            "type": "choice",
            "question": "Which ending most plausibly continues the text?",
            "options": options,
            "label": label,
            "source": "hellaswag",
        })
    return records


def preset_openbookqa(split: str, limit: Optional[int]) -> List[Dict[str, Any]]:
    """4지선다 과학 (Apache 2.0). ARC 와 가까운 held-out 대조군."""
    dataset = _load_hf("allenai/openbookqa", "main", split)
    records: List[Dict[str, Any]] = []
    for i, row in enumerate(dataset):
        if limit is not None and i >= limit:
            break
        labels = list(row["choices"]["label"])
        texts = list(row["choices"]["text"])
        options = {label: text for label, text in zip(labels, texts)}
        answer = str(row["answerKey"])
        if answer not in options:
            continue
        records.append({
            "state": row["question_stem"],
            "type": "choice",
            "question": "Which option is the correct answer?",
            "options": options,
            "label": answer,
            "source": "openbookqa",
        })
    return records


_PRESETS = {
    "openbookqa": {"fn": preset_openbookqa, "train": "train", "val": "validation"},
    "commonsense_qa": {"fn": preset_commonsense_qa, "train": "train", "val": "validation"},
    "hellaswag": {"fn": preset_hellaswag, "train": "train", "val": "validation"},
}




def main() -> None:
    parser = argparse.ArgumentParser(description="build JSONL from HuggingFace presets")
    sub = parser.add_subparsers(dest="command", required=True)
    p_hf = sub.add_parser("hf")
    p_hf.add_argument("--preset", action="append", required=True, choices=sorted(_PRESETS.keys()))
    p_hf.add_argument("--out", required=True)
    p_hf.add_argument("--limit", type=int, default=None)
    p_hf.add_argument("--val-limit", type=int, default=2000)
    p_hf.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    train_records: List[Dict[str, Any]] = []
    val_records: List[Dict[str, Any]] = []
    for preset in args.preset:
        spec = _PRESETS[preset]
        print(f"loading {preset} ...")
        train_records.extend(spec["fn"](spec["train"], args.limit))
        val_records.extend(spec["fn"](spec["val"], args.val_limit))
    random.Random(args.seed).shuffle(train_records)
    random.Random(args.seed + 1).shuffle(val_records)
    write_jsonl(os.path.join(args.out, "train.jsonl"), train_records)
    write_jsonl(os.path.join(args.out, "val.jsonl"), val_records)
    print(f"wrote {len(train_records)} -> {args.out}/train.jsonl")
    print(f"wrote {len(val_records)} -> {args.out}/val.jsonl")


if __name__ == "__main__":
    main()
