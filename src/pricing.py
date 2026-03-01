import json
from pathlib import Path

def load_prices(path="/Users/arcanainc/Desktop/Pytorch/ProjectDL/prices/prices.json"):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def estimate_total(predicted_items, prices_path="/Users/arcanainc/Desktop/Pytorch/ProjectDL/prices/prices.json"):
    prices = load_prices(prices_path)
    total = 0.0
    breakdown = []
    for it in predicted_items:
        label = it["label"]
        w = float(it.get("weight_kg", 1.0))
        price_per_kg = float(prices[label]["price_kzt"])
        cost = w * price_per_kg
        breakdown.append({"label": label, "weight_kg": w, "price_per_kg": price_per_kg, "cost_kzt": cost})
        total += cost
    return total, breakdown

if __name__ == "__main__":
    test = [{"label": "apple", "weight_kg": 1.0}, {"label": "banana", "weight_kg": 1.5}]
    total, breakdown = estimate_total(test)
    print("Breakdown:", breakdown)
    print("TOTAL KZT:", total)