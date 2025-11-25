import os
import json
import re
import pandas as pd
import statsmodels.api as sm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
XLSX_PATH = os.path.join(BASE_DIR, '..', 'data', 'dashboard_raw_data.xlsx')

county_names = [
    "California", "Alameda", "Alpine", "Amador", "Butte", "Calaveras", "Colusa", "Contra Costa",
    "Del Norte", "El Dorado", "Fresno", "Glenn", "Humboldt", "Imperial", "Inyo",
    "Kern", "Kings", "Lake", "Lassen", "Los Angeles", "Madera", "Marin", "Mariposa",
    "Mendocino", "Merced", "Modoc", "Mono", "Monterey", "Napa", "Nevada", "Orange",
    "Placer", "Plumas", "Riverside", "Sacramento", "San Benito", "San Bernardino",
    "San Diego", "San Francisco", "San Joaquin", "San Luis Obispo", "San Mateo",
    "Santa Barbara", "Santa Clara", "Santa Cruz", "Shasta", "Sierra", "Siskiyou",
    "Solano", "Sonoma", "Stanislaus", "Sutter", "Tehama", "Trinity", "Tulare",
    "Tuolumne", "Ventura", "Yolo", "Yuba"
]

sheets_to_use = ['Employment Rate', 'Wage Progression', 'PostCWEmployment', 'Exits With Earnings', 'Reentry', 'Reentry After Exit with Earning']

def load_tables(max_col=15, select_sheets=None):
    sheets = sheets_to_use
    if select_sheets:
        sheets = [s for s in sheets if s in select_sheets]

    indicator_tables = {}
    for sheet in sheets:
        raw_df = pd.read_excel(
            XLSX_PATH,
            sheet_name=sheet,
            header=None,
            engine='openpyxl'
        )
        raw_df = raw_df.dropna(how='all')
        header = raw_df.iloc[1].to_list()
        df = raw_df[2:-1].copy()
        df.columns = header
        indicator = sheet.strip().lower().replace(' ', '_')
        indicator_tables[indicator] = df

    return indicator_tables

def split_by_county(indicator_tables):
    structured_data = {}
    for indicator, df in indicator_tables.items():
        df.columns = [str(c).strip().lower() for c in df.columns]
        structured_data[indicator] = {}

        for county in county_names:
            county_df = df[df["county"] == county]
            structured_data[indicator][county] = county_df
    return structured_data


def to_group(structured_data):
    grouped_data = {}
    for indicator, county_data in structured_data.items():
        grouped_data[indicator] = {}
        for county, df in county_data.items():
            if df is None:
                continue
            df.columns = [str(c).strip().lower() for c in df.columns]

            category = "category"
            subcategory = "subcategory"
            if category is None or subcategory is None:
                continue
            grouped_data[indicator][county] = {}
            groups = df.groupby([category, subcategory])
            for (cat, subcat), group_df in groups:
                if cat not in grouped_data[indicator][county]:
                    grouped_data[indicator][county][cat] = {}
                grouped_data[indicator][county][cat][subcat] = group_df.copy()
    return grouped_data



def detect_outcome_col(indicator):
    if indicator == 'wage_progression':
        return 'med_earns'
    elif indicator == 'exits_with_earnings':
        return 'earn_rate'
    elif indicator == 'reentry_after_exit_with_earning':
        return 'reents_aewe_rate'
    else:
        return 'rate'



def compute_group_stats(df_group, indicator):
    df = df_group.copy()
    df.columns = [str(c).strip().lower().replace(' ', '_') for c in df.columns]
    outcome_col = detect_outcome_col(indicator)

    df[outcome_col] = df[outcome_col].astype(str)
    df[outcome_col] = df[outcome_col].str.replace('%', '', regex=False)
    df[outcome_col] = df[outcome_col].str.replace(',', '', regex=False)
    df[outcome_col] = df[outcome_col].str.replace('*', '', regex=False)
    df[outcome_col] = pd.to_numeric(df[outcome_col], errors='coerce')
    outcome = df[outcome_col].dropna()
    if outcome.empty:
        return None

    if indicator in ('reentry', 'reentry_after_exit_with_earnings'):
        df["reentry_year"] = (
            df["reentry_year"]
            .astype(str)
            .str.extract(r"(\d{4})")
            .astype(int)
        )
        print("before sort:", df['reentry_year'])
        df = df.dropna(subset=["reentry_year"])
        df = df[df['reentry_year']>=2017]
        df = df.sort_values(by='reentry_year')
        print("after sort:", df['reentry_year'])
        df['time_index'] = df['reentry_year']
    else:
        df["year"] = (
            df["year"]
            .astype(str)
            .str.extract(r"(\d{4})")
            .astype(int)
        )
        print("before sort:", df['year'])
        df = df.dropna(subset=["year"])
        df = df[df['year']>=2017]
        df = df.sort_values(by='year')
        print("after sort:", df['year'])
        df['time_index'] = df['year']
    df['time_index'] = df['time_index'] - df['time_index'].min()
    print('time_index:', df['time_index'])
    valid = df[[outcome_col, 'time_index']].dropna()
    if valid.empty:
        return None

    earliest = outcome.iloc[0]
    latest = outcome.iloc[-1]
    change = latest - earliest

    stats = {
        "mean": float(outcome.mean()),
        "median": float(outcome.median()),
        "std": float(outcome.std()),
        "max": float(outcome.max()),
        "min": float(outcome.min()),
        "earliest": float(earliest),
        "latest": float(latest),
        "change": float(change),
        "samples": int(len(outcome)),
    }

    df_yearly = df.groupby("time_index")[outcome_col].mean().reset_index()
    print('yearly mean', df_yearly)
    print('time index', df['time_index'])
    if df_yearly.shape[0] < 2:
        stats["trend"] = None
    else:
        X = sm.add_constant(df_yearly["time_index"])
        y = df_yearly[outcome_col]

        try:
            model = sm.OLS(y, X).fit()
            stats["trend"] = float(model.params["time_index"])
        except:
            stats["trend"] = None

    return stats



def compute_all_stats(grouped_data):
    all_stats = {}

    for indicator, county_dict in grouped_data.items():
        print("\n=== PROCESSING INDICATOR:", indicator, "===")
        all_stats[indicator] = {}

        for county, cat_dict in county_dict.items():
            print("  -> COUNTY:", county)
            all_stats[indicator][county] = {}

            for category, subcat_dict in cat_dict.items():
                all_stats[indicator][county][category] = {}

                for subcat, df_group in subcat_dict.items():
                    stats = compute_group_stats(df_group, indicator)
                    if stats:
                        all_stats[indicator][county][category][subcat] = stats

    return all_stats



def save_stats_json(all_stats, path="all_stats.json"):
    with open(path, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"Saved stats JSON to {path}")

# tables = load_tables()
# structured_data = split_by_county(tables)
# grouped_data = to_group(structured_data)
# all_stats = compute_all_stats(grouped_data)
# save_stats_json(all_stats, os.path.join(BASE_DIR, "..", "data", "dashboard_quant_stats.json"))

if __name__ == '__main__':
    tables = load_tables()
    structured_data = split_by_county(tables)
    grouped_data = to_group(structured_data)
    all_stats = compute_all_stats(grouped_data)
    save_stats_json(all_stats, os.path.join(BASE_DIR, "..", "data", "dashboard_quant_stats.json"))





