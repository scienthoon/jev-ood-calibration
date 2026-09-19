"""
scripts/eval_dump.py — accuracy and calibration metrics from a prediction dump. Model-agnostic.

Dump record (one JSON per line):
  {"type", "option_keys", "probs", "target", "pred", "gold", "source"?, "confidence"?, "error"?}
  Rows with "error" are excluded and only counted.

    python -m scripts.eval_dump --dump results/jev_synth.jsonl
    python -m scripts.eval_dump --dump results/jev_synth.jsonl --fit-temperature          # sign/size of miscalibration
    python -m scripts.eval_dump --dump results/jev_synth.jsonl --confidence-reliability   # the provider's own confidence field
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Any, Dict, List

import torch

from .metrics import fit_temperature, format_reliability_table, reliability, summarize


def load_dump(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def to_tensors(rows: List[Dict[str, Any]], eps: float):
    k_max = max(len(row["probs"]) for row in rows)
    n = len(rows)
    logits = torch.full((n, k_max), float("-inf"))
    target = torch.zeros((n, k_max))
    mask = torch.zeros((n, k_max), dtype=torch.bool)
    for i, row in enumerate(rows):
        probs = torch.tensor([float(p) for p in row["probs"]])
        k = probs.numel()
        logits[i, :k] = torch.log(probs.clamp(min=eps))
        target[i, :k] = torch.tensor([float(t) for t in row["target"]])
        mask[i, :k] = True
    return logits, target, mask


def ece_noise_floor(logits: torch.Tensor, mask: torch.Tensor, n_bins: int = 15, n_sims: int = 200, seed: int = 0) -> Dict[str, float]:
    """
    ECE 의 노이즈 바닥. 예측 분포는 그대로 두고 라벨을 그 분포에서 샘플링하면
    "완벽히 캘리브레이션된 모델" 이 되는데, 유한한 n 과 bin 때문에 그 모델도 ECE 가 0 이 아니다.
    그 값의 평균/표준편차를 돌려준다. 측정된 ECE 를 이 바닥과 비교해야 "구분 가능한 오차" 인지 알 수 있다.
    """
    generator = torch.Generator().manual_seed(seed)
    probs = torch.softmax(logits.masked_fill(~mask, float("-inf")), dim=-1)
    n, k = probs.shape
    values: List[float] = []
    for _ in range(n_sims):
        sampled = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(1)
        target = torch.nn.functional.one_hot(sampled, k).float() * mask
        values.append(float(reliability(logits, target, mask, n_bins=n_bins)["ece"]))
    t = torch.tensor(values)
    return {"mean": float(t.mean()), "std": float(t.std()), "n_sims": n_sims}


def confidence_reliability(rows: List[Dict[str, Any]], n_bins: int = 15) -> Dict[str, Any]:
    """
    TypeSafe 가 따로 주는 confidence 필드가 정답률을 예측하는지 본다.
    confidence 를 bin 으로 나누고 각 bin 의 정답률(pred == gold)을 잰다.
    max-prob 기반 ECE 와는 별개의 질문: "이 confidence 통계를 임계값으로 써도 되는가".
    """
    pairs = []
    for row in rows:
        c = row.get("confidence")
        if c is None:
            continue
        try:
            c = float(c)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(c):
            continue
        pairs.append((c, 1.0 if row["pred"] == row["gold"] else 0.0))
    if not pairs:
        return {"n": 0, "bins": []}
    confidence = torch.tensor([p[0] for p in pairs])
    correct = torch.tensor([p[1] for p in pairs])
    low = float(confidence.min())
    high = float(confidence.max())
    if high <= low:
        high = low + 1e-6
    edges = torch.linspace(low, high, n_bins + 1)
    bins = []
    ece = 0.0
    for b in range(n_bins):
        lower = edges[b]
        upper = edges[b + 1]
        in_bin = (confidence >= lower) & (confidence <= upper) if b == 0 else (confidence > lower) & (confidence <= upper)
        count = int(in_bin.sum())
        if count == 0:
            continue
        avg_conf = float(confidence[in_bin].mean())
        avg_acc = float(correct[in_bin].mean())
        ece += abs(avg_conf - avg_acc) * count / len(pairs)
        bins.append({"lower": float(lower), "upper": float(upper), "count": count, "avg_conf": avg_conf, "avg_acc": avg_acc})
    return {"n": len(pairs), "range": [low, high], "ece_if_probability": ece, "bins": bins}


def main() -> None:
    parser = argparse.ArgumentParser(description="metrics from a prediction dump")
    parser.add_argument("--dump", required=True)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--fit-temperature", action="store_true")
    parser.add_argument("--confidence-reliability", action="store_true", help="reliability of the dump's own confidence field (e.g. TypeSafe)")
    args = parser.parse_args()

    all_rows = load_dump(args.dump)
    rows = [row for row in all_rows if "error" not in row and row.get("probs")]
    errors = len(all_rows) - len(rows)
    print(f"dump: {args.dump}  rows: {len(rows)}  errors: {errors}")
    if not rows:
        return

    logits, target, mask = to_tensors(rows, args.eps)
    types = [row["type"] for row in rows]
    sources = [str(row.get("source") or "") for row in rows]

    print("overall:", json.dumps(summarize(logits, target, mask, temperature=1.0)))
    for qtype in sorted(set(types)):
        index = torch.tensor([i for i, t in enumerate(types) if t == qtype])
        print(f"  {qtype}: {json.dumps(summarize(logits[index], target[index], mask[index], temperature=1.0))}")
    for source in sorted(set(s for s in sources if s)):
        index = torch.tensor([i for i, s in enumerate(sources) if s == source])
        print(f"  [{source}]: {json.dumps(summarize(logits[index], target[index], mask[index], temperature=1.0))}")
    print("reliability (max-prob):")
    print(format_reliability_table(reliability(logits, target, mask, temperature=1.0)))
    measured = float(reliability(logits, target, mask, temperature=1.0)["ece"])
    floor = ece_noise_floor(logits, mask)
    ratio = measured / floor["mean"] if floor["mean"] > 0 else float("inf")
    print(f"ECE noise floor (perfectly calibrated model, same predictions, n={len(rows)}): {floor['mean']:.4f} ± {floor['std']:.4f}  ->  measured/floor = {ratio:.2f}x")
    for source in sorted(set(s for s in sources if s)):
        index = torch.tensor([i for i, s in enumerate(sources) if s == source])
        m = float(reliability(logits[index], target[index], mask[index])["ece"]); f = ece_noise_floor(logits[index], mask[index])
        print(f"  [{source}] ECE {m:.4f} / floor {f['mean']:.4f} = {m / f['mean'] if f['mean'] > 0 else float('inf'):.2f}x")

    if args.fit_temperature:
        fitted = fit_temperature(logits, target, mask)
        print(f"\nrefit temperature: T={fitted:.4f}  (1.0 = already calibrated, >1 overconfident, <1 underconfident)")
        print("after:", json.dumps(summarize(logits, target, mask, temperature=fitted)))
        for source in sorted(set(s for s in sources if s)):
            index = torch.tensor([i for i, s in enumerate(sources) if s == source])
            print(f"  [{source}]: {json.dumps(summarize(logits[index], target[index], mask[index], temperature=fitted))}")
        print(format_reliability_table(reliability(logits, target, mask, temperature=fitted)))

    if args.confidence_reliability:
        rel = confidence_reliability(rows)
        print(f"\nconfidence field reliability: n={rel['n']}")
        if rel["n"] > 0:
            print(f"  range: {rel['range'][0]:.3f} .. {rel['range'][1]:.3f}   ECE-if-read-as-probability: {rel['ece_if_probability']:.4f}")
            print("  bin              count   avg_conf   avg_acc")
            for b in rel["bins"]:
                print(f"  {b['lower']:.3f}-{b['upper']:.3f}   {b['count']:6d}   {b['avg_conf']:8.3f}   {b['avg_acc']:7.3f}")


if __name__ == "__main__":
    main()
