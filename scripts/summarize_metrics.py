from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import METRICS_DIR


def main() -> None:
    rows = []
    required_keys = {"accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"}
    for path in sorted(METRICS_DIR.glob("*.json")):
        metrics = json.loads(path.read_text(encoding="utf-8"))
        if not required_keys.issubset(metrics):
            continue

        rows.append(
            {
                "model": path.stem,
                "accuracy": metrics["accuracy"],
                "macro_precision": metrics["macro_precision"],
                "macro_recall": metrics["macro_recall"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
            }
        )

    if not rows:
        raise FileNotFoundError("No metric JSON files found. Train models first.")

    output_path = METRICS_DIR / "summary.csv"
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    for row in sorted(rows, key=lambda item: item["macro_f1"], reverse=True):
        print(
            f"{row['model']}: macro_f1={row['macro_f1']:.4f}, "
            f"accuracy={row['accuracy']:.4f}, weighted_f1={row['weighted_f1']:.4f}"
        )
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
