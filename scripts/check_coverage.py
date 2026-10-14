"""Check how many benchmark questions have ground-truth docs in our 10K index.

For each question with expected_doc_ids, query ChromaDB for chunks whose
doc_id matches. Count questions with at least one expected doc present.

This tells us whether our 10K slice can meaningfully evaluate the 500-question
benchmark. If coverage is low, the retrieval metrics are measuring sampling
luck, not retrieval quality.
"""

from __future__ import annotations

from collections import Counter

from datasets import load_dataset

from src.ingestion.store import VectorStore


def main() -> None:
    print("Loading benchmark questions...")
    questions = load_dataset(
        "onyx-dot-app/EnterpriseRAG-Bench",
        "questions",
        split="test",
    )

    print("Opening 10K index...")
    store = VectorStore()
    collection = store._collection

    total_with_gt = 0
    total_with_hit = 0
    hits_by_type: Counter[str] = Counter()
    total_by_type: Counter[str] = Counter()

    print()
    print("Checking coverage per question...")
    for i, row in enumerate(questions, 1):
        if i % 50 == 0:
            print(f"  {i}/{len(questions)}  "
                  f"({total_with_hit}/{total_with_gt} so far)")

        expected = row["expected_doc_ids"] or []
        if not expected:
            continue

        total_with_gt += 1
        q_type = row["question_type"]
        total_by_type[q_type] += 1

        hit = False
        for doc_id in expected:
            result = collection.get(where={"doc_id": doc_id})
            if result["ids"]:
                hit = True
                break

        if hit:
            total_with_hit += 1
            hits_by_type[q_type] += 1

    print()
    print("=" * 60)
    print("COVERAGE REPORT")
    print("=" * 60)
    print(f"Total questions:               {len(questions)}")
    print(f"Questions with ground truth:   {total_with_gt}")
    print(f"Questions with >= 1 hit:       {total_with_hit}")
    coverage = 100 * total_with_hit / max(1, total_with_gt)
    print(f"Coverage:                      {coverage:.1f}%")
    print()
    print("Coverage by question type:")
    for q_type in sorted(total_by_type.keys()):
        t = total_by_type[q_type]
        h = hits_by_type[q_type]
        pct = 100 * h / max(1, t)
        print(f"  {q_type:32s}  {h:3d}/{t:3d}  ({pct:5.1f}%)")
    print()
    print("If coverage is very low (<20%), the 10K slice cannot evaluate")
    print("this benchmark, and we need to rebuild at a larger corpus size.")


if __name__ == "__main__":
    main()