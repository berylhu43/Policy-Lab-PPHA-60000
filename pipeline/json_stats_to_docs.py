import json
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def build_summary_text(indicator, county, category, subcat, stats):

    name = indicator.replace('_', ' ').title()

    trend = stats.get("trend", None)
    if trend is None or pd.isna(trend):
        trend_text = "Trend is unclear."
    elif trend > 0:
        trend_text = f"Trend: increasing over time (slope ≈ +{trend:.4f})."
    elif trend < 0:
        trend_text = f"Trend: decreasing over time (slope ≈ {trend:.4f})."
    else:
        trend_text = "Trend is flat (slope = 0)."

    text = (
        f"{name} – {county} – {category} – {subcat}\n"
        f"Latest value: {stats['latest']:.3f}, change from earliest: {stats['change']:.3f}.\n"
        f"Average: {stats['mean']:.3f}, median: {stats['median']:.3f}, "
        f"max: {stats['max']:.3f}, min: {stats['min']:.3f}.\n"
        f"Samples: {stats['samples']}.\n"
        f"{trend_text}"
    )

    return text

def stats_json_to_documents(stats_json_path, output_path):

    with open(stats_json_path, "r") as f:
        all_stats = json.load(f)

    docs = []

    for indicator, county_dict in all_stats.items():
        for county, cat_dict in county_dict.items():
            for category, subcat_dict in cat_dict.items():
                for subcat, stats in subcat_dict.items():

                    doc_id = f"{indicator}|{county}|{category}|{subcat}"

                    text = build_summary_text(
                        indicator, county, category, subcat, stats
                    )

                    metadata = {
                        "indicator": indicator,
                        "county": county,
                        "category": category,
                        "subcategory": subcat,
                        "type": "quant"
                    }

                    docs.append({
                        "id": doc_id,
                        "text": text,
                        "metadata": metadata
                    })

    with open(output_path, "w") as f:
        json.dump(docs, f, indent=2)

    print(f"Saved {len(docs)} embedding docs to {output_path}")

if __name__ == "__main__":
    stats_path = os.path.join(BASE_DIR, "..", "data", "dashboard_quant_stats.json")
    docs_path = os.path.join(BASE_DIR, "..", "data", "dashboard_quant_docs.json")
    stats_json_to_documents(stats_path, docs_path)