"""
data/generate.py — the synthetic support-ticket set used in this study. Rule-based, no LLM.

    python data/generate.py --out data --n 3000 --seed 0       # reproduces data/val.jsonl (900 records)
    python data/generate.py --out data_fresh --n 3000 --seed 7 # a new, guaranteed-uncontaminated set

Each ticket is {channel, customer_tier, subject, body}: 20 templates across 4 queues, random order numbers,
amounts, days, and a calm or angry closing line (30% angry; some lines in Korean). Three questions per ticket:
  queue    (choice, 4)  = the template's queue
  priority (score, 4)   = template base urgency (0-2) + 1 if angry + 1 if tier in {gold, enterprise}, clipped to 0..3
  angry    (noul)       = whether an angry closing line was appended
5% of labels are randomly corrupted (label_noise) to mimic operational data, so no model can exceed ~95%.
Records use the jevlocal JSONL schema: {state, type, question, options|levels, label}.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any, Dict, List, Sequence, Tuple


_SYNTH_TEMPLATES: Dict[str, List[Dict[str, Any]]] = {
    "billing": [
        {"subject": "Charged twice for order #{order}", "body": "My card shows two charges of ${amount} for one order. {tail}", "urgency": 2},
        {"subject": "Refund not received", "body": "I returned the item {days} days ago and the refund for ${amount} is still missing. {tail}", "urgency": 1},
        {"subject": "Wrong amount on invoice", "body": "Invoice #{order} says ${amount} but the quote was lower. {tail}", "urgency": 1},
        {"subject": "구독 요금이 두 번 결제됐어요", "body": "이번 달 카드 명세서에 {amount}달러가 두 번 찍혀 있습니다. {tail}", "urgency": 2},
        {"subject": "환불이 아직 안 들어왔습니다", "body": "{days}일 전에 반품했는데 환불이 안 됐어요. {tail}", "urgency": 1},
    ],
    "shipping": [
        {"subject": "Package marked delivered but not here", "body": "Tracking for order #{order} says delivered {days} days ago. Nothing arrived. {tail}", "urgency": 2},
        {"subject": "Where is my order?", "body": "Order #{order} was placed {days} days ago and tracking has not updated. {tail}", "urgency": 1},
        {"subject": "Damaged on arrival", "body": "The box for order #{order} was crushed and the item inside is broken. {tail}", "urgency": 2},
        {"subject": "배송이 너무 늦어요", "body": "주문 #{order} 한 지 {days}일이 지났는데 아직 배송 준비중입니다. {tail}", "urgency": 1},
        {"subject": "배송 완료라는데 못 받았습니다", "body": "송장은 배송 완료인데 집에 아무것도 없어요. {tail}", "urgency": 2},
    ],
    "technical": [
        {"subject": "App crashes on login", "body": "Since the last update the app closes immediately after I enter my password. {tail}", "urgency": 2},
        {"subject": "Cannot reset password", "body": "The reset email for my account never arrives, checked spam too. {tail}", "urgency": 1},
        {"subject": "Export button does nothing", "body": "Clicking export on the reports page shows a spinner forever. {tail}", "urgency": 1},
        {"subject": "로그인이 안 됩니다", "body": "비밀번호를 맞게 입력해도 오류 코드 {order}가 뜹니다. {tail}", "urgency": 2},
        {"subject": "앱이 계속 꺼져요", "body": "업데이트 이후 사진을 열면 앱이 종료됩니다. {tail}", "urgency": 1},
    ],
    "general": [
        {"subject": "Question about your return policy", "body": "How many days do I have to return an unopened item? {tail}", "urgency": 0},
        {"subject": "Do you ship to Canada?", "body": "Planning to order as a gift, want to confirm shipping options first. {tail}", "urgency": 0},
        {"subject": "Feature suggestion", "body": "It would be nice to sort the dashboard by date. Not urgent. {tail}", "urgency": 0},
        {"subject": "영업시간 문의", "body": "고객센터 전화 상담은 몇 시까지인가요? {tail}", "urgency": 0},
        {"subject": "제품 사양 질문", "body": "이 모델이 해외 전압에서도 쓸 수 있는지 궁금합니다. {tail}", "urgency": 0},
    ],
}

_TAILS_CALM: List[str] = [
    "Thanks in advance.",
    "Let me know what you need from me.",
    "Appreciate any help.",
    "확인 부탁드립니다.",
    "답변 기다리겠습니다.",
    "",
]

_TAILS_ANGRY: List[str] = [
    "This is unacceptable and I want it fixed today.",
    "I have contacted you three times already. Ridiculous.",
    "If this is not resolved I am disputing the charge and leaving a review.",
    "정말 화가 납니다. 당장 처리해 주세요.",
    "이게 몇 번째인지 모르겠네요. 진짜 실망입니다.",
]

_QUEUE_OPTIONS: Dict[str, str] = {
    "billing": "Payments, refunds, duplicate charges",
    "shipping": "Delivery status, lost or damaged packages",
    "technical": "App or website bugs, login problems",
    "general": "Questions, feedback, anything else",
}

_PRIORITY_LEVELS: List[str] = ["Low", "Normal", "High", "Critical"]


def build_synthetic(n_states: int, label_noise: float, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    queues = list(_SYNTH_TEMPLATES.keys())
    records: List[Dict[str, Any]] = []
    for _ in range(n_states):
        queue = rng.choice(queues)
        template = rng.choice(_SYNTH_TEMPLATES[queue])
        angry = rng.random() < 0.3
        tail = rng.choice(_TAILS_ANGRY if angry else _TAILS_CALM)
        tier = rng.choice(["free", "standard", "gold", "enterprise"])
        channel = rng.choice(["email", "chat", "phone", "app"])
        fill = {
            "order": rng.randint(1000, 9999),
            "amount": rng.choice([19.99, 49.00, 89.99, 120.50, 300.00]),
            "days": rng.randint(1, 21),
            "tail": tail,
        }
        state = {
            "channel": channel,
            "customer_tier": tier,
            "subject": template["subject"].format(**fill),
            "body": template["body"].format(**fill).strip(),
        }

        # priority: 템플릿 기본 긴급도 + 화남 + 티어
        priority = int(template["urgency"])
        if angry:
            priority += 1
        if tier in ("gold", "enterprise") and priority >= 1:
            priority += 1
        priority = max(0, min(len(_PRIORITY_LEVELS) - 1, priority))

        queue_label = queue
        angry_label = angry
        if label_noise > 0.0:
            if rng.random() < label_noise:
                queue_label = rng.choice([q for q in queues if q != queue])
            if rng.random() < label_noise:
                priority = rng.randint(0, len(_PRIORITY_LEVELS) - 1)
            if rng.random() < label_noise:
                angry_label = not angry

        records.append({
            "state": state,
            "type": "choice",
            "question": "Which support queue should handle this ticket?",
            "options": dict(_QUEUE_OPTIONS),
            "label": queue_label,
        })
        records.append({
            "state": state,
            "type": "score",
            "question": "How should this ticket be prioritized?",
            "levels": list(_PRIORITY_LEVELS),
            "label": priority,
        })
        records.append({
            "state": state,
            "type": "noul",
            "question": "The customer sounds angry.",
            "label": angry_label,
        })
    return records


def split_records(records: Sequence[Dict[str, Any]], val_fraction: float, seed: int = 0) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    n_val = int(round(len(shuffled) * val_fraction))
    return shuffled[n_val:], shuffled[:n_val]


def write_jsonl(path: str, records: Sequence[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="generate the synthetic support-ticket set")
    parser.add_argument("--out", required=True)
    parser.add_argument("--n", type=int, default=3000, help="number of tickets (records = x3)")
    parser.add_argument("--label-noise", type=float, default=0.05)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    records = build_synthetic(args.n, args.label_noise, args.seed)
    # keep the 3 records of one ticket on the same side of the split
    groups = [records[i:i + 3] for i in range(0, len(records), 3)]
    train_groups, val_groups = split_records(groups, args.val_fraction, seed=args.seed)  # type: ignore[arg-type]
    train_records = [r for g in train_groups for r in g]
    val_records = [r for g in val_groups for r in g]
    write_jsonl(os.path.join(args.out, "train.jsonl"), train_records)
    write_jsonl(os.path.join(args.out, "val.jsonl"), val_records)
    print(f"wrote {len(train_records)} -> {args.out}/train.jsonl")
    print(f"wrote {len(val_records)} -> {args.out}/val.jsonl")


if __name__ == "__main__":
    main()
