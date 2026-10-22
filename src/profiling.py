import json

from src.config import PROCESSED_DIR, PROFILING_PATH
from src.data_loader import load_all, unique_articles


def _blank(series):
    return series.isna() | (series.str.strip() == "")


def profile(df):
    uniq = unique_articles(df)
    return {
        "total_rows": int(len(df)),
        "rows_per_clipping": {
            k: int(v) for k, v in df["clipping_name"].value_counts().items()
        },
        "unique_original_id": int(df["original_id"].nunique()),
        "duplicate_rows": int(len(df) - df["original_id"].nunique()),
        "unique_title_body": int(df.drop_duplicates(["title", "body"]).shape[0]),
        "unique_body": int(df["body"].nunique()),
        "missing_title": int(_blank(df["title"]).sum()),
        "missing_body": int(_blank(df["body"]).sum()),
        "entities_empty_pct": round(float(_blank(df["entities"]).mean() * 100), 2),
        "language_unique_articles": {
            k: int(v) for k, v in uniq["language"].value_counts().items()
        },
        "date_min": str(df["date_published"].min()),
        "date_max": str(df["date_published"].max()),
        "unparsed_dates": int(df["date_published"].isna().sum()),
    }

def main():
    report = profile(load_all())
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    PROFILING_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("Tersimpan:", PROFILING_PATH)


if __name__ == "__main__":
    main()