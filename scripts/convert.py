"""
scripts/convert.py — build jevlocal-schema JSONL from public HuggingFace datasets.

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


def _label_names(dataset, column: str, fallback: Sequence[str]) -> List[str]:
    try:
        return list(dataset.features[column].names)
    except Exception:
        return list(fallback)


def preset_klue_ynat(split: str, limit: Optional[int]) -> List[Dict[str, Any]]:
    # 구 "klue" 는 스크립트 기반이라 datasets>=3 에서 로드가 거부된다. parquet 로 옮겨진 "klue/klue" 를 쓴다.
    dataset = _load_hf("klue/klue", "ynat", split)
    names = _label_names(dataset, "label", ["IT과학", "경제", "사회", "생활문화", "세계", "스포츠", "정치"])
    options = {name: name for name in names}
    records: List[Dict[str, Any]] = []
    for i, row in enumerate(dataset):
        if limit is not None and i >= limit:
            break
        records.append({
            "state": row["title"],
            "type": "choice",
            "question": "이 뉴스 제목의 주제 분류는 무엇인가?",
            "options": options,
            "label": names[int(row["label"])],
            "source": "klue_ynat",
        })
    return records


_NSMC_URLS = {
    "train": "https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt",
    "test": "https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt",
}


def _nsmc_rows(split: str):
    """
    NSMC 는 HF 허브에서 스크립트 기반이라 datasets>=3 에서 로드되지 않는다.
    원본 GitHub 의 TSV (id, document, label) 를 직접 받아 ~/.cache/jevlocal/nsmc/ 에 캐시한다.
    """
    import urllib.request
    url = _NSMC_URLS[split]
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "jevlocal", "nsmc")
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, os.path.basename(url))
    if not os.path.exists(path):
        print(f"downloading {url} ...")
        urllib.request.urlretrieve(url, path)
    with open(path, "r", encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != len(header):
                continue
            yield dict(zip(header, parts))


def preset_nsmc(split: str, limit: Optional[int]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for i, row in enumerate(_nsmc_rows(split)):
        if limit is not None and i >= limit:
            break
        document = row["document"].strip()
        if not document:
            continue
        records.append({
            "state": document,
            "type": "noul",
            "question": "이 영화 리뷰는 긍정적이다.",
            "label": bool(int(row["label"]) == 1),
            "source": "nsmc",
        })
    return records


def preset_boolq(split: str, limit: Optional[int]) -> List[Dict[str, Any]]:
    dataset = _load_hf("google/boolq", None, split)
    records: List[Dict[str, Any]] = []
    for i, row in enumerate(dataset):
        if limit is not None and i >= limit:
            break
        records.append({
            "state": row["passage"],
            "type": "noul",
            "question": row["question"].rstrip("?") + "?",
            "label": bool(row["answer"]),
            "source": "boolq",
        })
    return records


def preset_arc_easy(split: str, limit: Optional[int]) -> List[Dict[str, Any]]:
    dataset = _load_hf("allenai/ai2_arc", "ARC-Easy", split)
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
            "source": "arc_easy",
        })
    return records


_BANKING77_URLS = {
    "train": "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv",
    "test": "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv",
}


def _banking77_rows(split: str) -> List[Dict[str, str]]:
    """
    HF 의 PolyAI/banking77 은 스크립트 기반이라 datasets>=3 에서 로드되지 않는다.
    원본 GitHub CSV (text, category) 를 받아 ~/.cache/jevlocal/banking77/ 에 캐시한다. CC BY 4.0.
    """
    import urllib.request
    url = _BANKING77_URLS[split]
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "jevlocal", "banking77")
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{split}.csv")
    if not os.path.exists(path):
        print(f"downloading {url} ...")
        urllib.request.urlretrieve(url, path)
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def preset_banking77(split: str, limit: Optional[int]) -> List[Dict[str, Any]]:
    """온라인 뱅킹 문의 77개 의도 (CC BY 4.0). 닫힌 라벨 집합 분류, 선택지 77개."""
    rows = _banking77_rows(split)
    # 라벨 집합은 train 기준으로 고정 (test 도 같은 77개). 정렬해서 순서를 결정적으로.
    names = sorted({row["category"] for row in _banking77_rows("train")})
    if len(names) != 77:
        raise SystemExit(f"banking77: expected 77 categories, found {len(names)}")
    options = {name: name.replace("_", " ") for name in names}
    records: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        if limit is not None and i >= limit:
            break
        if row["category"] not in options:
            continue
        records.append({
            "state": row["text"],
            "type": "choice",
            "question": "Which banking intent does this customer message express?",
            "options": options,
            "label": row["category"],
            "source": "banking77",
        })
    return records


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
    "arc_easy": {"fn": preset_arc_easy, "train": "train", "val": "validation"},
    "boolq": {"fn": preset_boolq, "train": "train", "val": "validation"},
}




def main() -> None:
    parser = argparse.ArgumentParser(description="build jevlocal JSONL from HuggingFace presets")
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
