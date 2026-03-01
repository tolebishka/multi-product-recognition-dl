from pathlib import Path
import json
import csv

RUNS_DIR = Path("/Users/arcanainc/Desktop/Pytorch/ProjectDL/runs")

rows = []
for run in RUNS_DIR.iterdir():
    if not run.is_dir():
        continue
    mpath = run / "metrics.json"
    if not mpath.exists():
        continue
    m = json.loads(mpath.read_text(encoding="utf-8"))
    rows.append({
        "run": run.name,
        "model": m["model"],
        "test_top1": round(m["test_top1"], 4),
        "test_top5": round(m["test_top5"], 4),
        "macro_f1": round(m["macro_f1"], 4),
        "weighted_f1": round(m["weighted_f1"], 4),
        "ms_per_image": round(m["speed"]["ms_per_image"], 2),
        "fps": round(m["speed"]["fps"], 1),
        "model_size_mb": round(m["model_size_mb"], 1),
    })

rows = sorted(rows, key=lambda r: r["test_top1"], reverse=True)

out_csv = RUNS_DIR / "summary.csv"
with out_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print("Saved:", out_csv)
print("Top results:")
for r in rows[:10]:
    print(r)