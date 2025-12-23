import argparse
import json
import statistics


def load_jsonl(path: str):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                raise SystemExit(f"Invalid JSON on line {line_no} in {path}")
    return items


def build_map(items, key="id"):
    out = {}
    for it in items:
        if key not in it:
            continue
        out[str(it[key])] = it
    return out


def extract_ranked_docids(pred_row):
    # Use citations order as a simple ranking proxy
    docids = []
    for c in pred_row.get("citations", []):
        d = c.get("docId")
        if d and d not in docids:
            docids.append(d)
    # Fallback: if team includes 'ranked_docIds' field, prefer it
    if pred_row.get("ranked_docIds"):
        ranked = []
        for d in pred_row["ranked_docIds"]:
            if d and d not in ranked:
                ranked.append(d)
        if ranked:
            return ranked
    return docids


def coverage_at_k(pred_map, gold_map, k: int):
    total = 0
    covered = 0
    for qid, g in gold_map.items():
        p = pred_map.get(qid)
        if not p:
            continue
        expected = set(g.get("expected_docIds", []))
        if not expected:
            continue
        total += 1
        ranked = extract_ranked_docids(p)[:k]
        if any(d in expected for d in ranked):
            covered += 1
    return covered / total if total else 0.0, total


def simple_accuracy(pred_map, gold_map):
    total = 0
    correct = 0
    for qid, g in gold_map.items():
        p = pred_map.get(qid)
        if not p:
            continue
        expected = set(g.get("expected_docIds", []))
        if not expected:
            continue
        total += 1
        ranked = extract_ranked_docids(p)
        top1 = ranked[0] if ranked else None
        if top1 in expected:
            correct += 1
    return correct / total if total else 0.0, total


def latency_stats(pred_map):
    lat = [p.get("latency_ms") for p in pred_map.values() if isinstance(p.get("latency_ms"), int)]
    if not lat:
        return {"n": 0, "avg_ms": None, "p95_ms": None}
    lat_sorted = sorted(lat)
    # p95 index
    idx = int(0.95 * (len(lat_sorted) - 1))
    return {
        "n": len(lat_sorted),
        "avg_ms": int(sum(lat_sorted) / len(lat_sorted)),
        "p95_ms": int(lat_sorted[idx]),
    }


def main():
    ap = argparse.ArgumentParser(description="MiniRAG evaluation (Iteration 2)")
    ap.add_argument("--pred", required=True, help="answers.jsonl produced by --batch")
    ap.add_argument("--gold", required=True, help="gold.jsonl with expected_docIds")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", required=True, help="Output report.json")
    args = ap.parse_args()

    pred_items = load_jsonl(args.pred)
    gold_items = load_jsonl(args.gold)

    pred = build_map(pred_items)
    gold = build_map(gold_items)

    cov, n_cov = coverage_at_k(pred, gold, args.k)
    acc, n_acc = simple_accuracy(pred, gold)
    lat = latency_stats(pred)

    report = {
        "coverage@k": {"k": args.k, "value": cov, "n": n_cov},
        "accuracy": {"value": acc, "n": n_acc},
        "latency": lat,
        "n_pred": len(pred),
        "n_gold": len(gold),
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
