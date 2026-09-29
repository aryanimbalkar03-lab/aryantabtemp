"""
prep.py — Credit Portfolio Risk & Operations Control Tower
==========================================================
Data preparation layer for the 6-dashboard Tableau suite built on the public
LendingClub accepted-loan tape (~2.26M loans, 2007–2018).

DESIGN PRINCIPLE (governance): this script does ONLY typing and cleaning.
Every business metric (PD, LGD, expected loss, vintage curves, pricing cushion,
HHI, stress deltas...) is defined ONCE in Tableau. Python recomputes them here
only to VALIDATE Tableau's numbers ("calc validation"), never to pre-bake them.

What it produces (all written to data/processed/):
  1. loans_clean.csv   -> primary Tableau source (one row per loan)
  2. dq_summary.csv    -> Dashboard 6 "Data Quality" source (one row per field)
  3. dictionary.csv    -> plain-English field dictionary (Dashboard 6 + README)
  4. status_map.csv    -> loan_status -> risk bucket mapping (definitions layer)
  5. reconciliation.txt-> printed totals that MUST match Tableau (controls gate)
  6. calc_validation.json -> python-side re-computation of every headline KPI

Usage:
  python prep/prep.py                       # auto-download via kagglehub
  python prep/prep.py --csv path/to/accepted_2007_to_2018Q4.csv
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
import zipfile
import numpy as np
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
KAGGLE_DATASET = "wordsforthewise/lending-club"
RAW_CSV_NAME = "accepted_2007_to_2018Q4.csv"

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Columns kept for the dashboards. Everything else is dropped so the Tableau
# extract stays lean (performance gate: hide/trim unused fields).
COLS = [
    "id", "loan_amnt", "funded_amnt", "term", "int_rate", "installment",
    "grade", "sub_grade", "home_ownership", "annual_inc",
    "verification_status", "issue_d", "loan_status", "purpose", "addr_state",
    "dti", "delinq_2yrs", "fico_range_low", "fico_range_high", "open_acc",
    "revol_util", "out_prncp", "total_pymnt", "total_rec_prncp",
    "total_rec_int", "total_rec_late_fee", "recoveries", "last_pymnt_d",
    "application_type",
]

# Business date used by the Mature Flag (portfolio cut-off = end of dataset).
MATURITY_CUTOFF = pd.Timestamp("2018-12-31")

# Status -> risk bucket (the single agreed definition, mirrored in Tableau).
STATUS_BUCKET = {
    "Fully Paid": "Closed - Paid in full",
    "Charged Off": "Closed - Loss",
    "Default": "Closed - Loss",
    "Does not meet the credit policy. Status:Fully Paid": "Closed - Policy (Paid)",
    "Does not meet the credit policy. Status:Charged Off": "Closed - Policy (Loss)",
    "Current": "Open - Performing",
    "In Grace Period": "Delinquent - Grace",
    "Late (16-30 days)": "Delinquent - 16-30",
    "Late (31-120 days)": "Delinquent - 31-120",
    "Late (more than 120 days)": "Delinquent - 120+",
}
DEFAULT_STATUSES = [
    "Charged Off", "Default",
    "Does not meet the credit policy. Status:Charged Off",
]
LIVE_STATUSES = [
    "Current", "In Grace Period", "Late (16-30 days)", "Late (31-120 days)",
]


# --------------------------------------------------------------------------
# Load
# --------------------------------------------------------------------------
def _find_in_dir(d: Path) -> Path | None:
    """Find the accepted-loan CSV anywhere under d (flat, nested or gzipped)."""
    hits = sorted(p for p in d.rglob("*")
                  if p.is_file() and p.name.lower().startswith(RAW_CSV_NAME.lower()))
    for p in hits:
        if p.suffix == ".csv":
            return p
    for p in hits:
        if p.suffix == ".gz":
            out = RAW_DIR / RAW_CSV_NAME
            if not out.exists():
                import gzip
                print(f"[prep] gunzipping {p} -> {out}")
                with gzip.open(p, "rb") as fi, open(out, "wb") as fo:
                    shutil.copyfileobj(fi, fo)
            return out
    return None


def resolve_raw_csv(csv_arg: str | None) -> Path:
    """Locate the raw LendingClub CSV (arg > local cache > kagglehub download)."""
    if csv_arg:
        return Path(csv_arg)

    # 1) already copied into data/raw ?
    local = _find_in_dir(RAW_DIR)
    if local:
        print(f"[prep] using local raw file: {local}")
        return local

    # 2) already in the kagglehub cache from a previous run ?
    cache_root = Path.home() / ".cache" / "kagglehub" / "datasets"
    owner, name = KAGGLE_DATASET.split("/")
    cached = cache_root / owner / name
    if cached.exists():
        found = _find_in_dir(cached)
        if found:
            print(f"[prep] using kagglehub cache: {found}")
            return found

    # 3) fresh download via kagglehub.
    try:
        import kagglehub  # noqa: PLC0415
    except ImportError:
        sys.exit("[prep] pip install kagglehub  (or pass --csv path)")

    print(f"[prep] downloading kaggle dataset: {KAGGLE_DATASET}")
    path = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    print("Path to dataset files:", path)

    found = _find_in_dir(path)
    if found is None:
        zips = list(path.rglob("*.zip"))
        if not zips:
            sys.exit(f"[prep] {RAW_CSV_NAME} not found under {path}")
        with zipfile.ZipFile(zips[0]) as zf:
            zf.extractall(path)
        found = _find_in_dir(path)
    if found is None:
        sys.exit(f"[prep] {RAW_CSV_NAME} not found under {path}")
    return found


# --------------------------------------------------------------------------
# Cleaning / typing ONLY (no metrics baked in)
# --------------------------------------------------------------------------
def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Typing + cleaning only. Columns arrive as pandas ArrowDtype (many of
    them dictionary-encoded by pyarrow), so every step below is written to be
    safe on that representation and to keep memory flat."""

    def cat(name: str):
        """Return a column as a plain pandas Categorical (cheap: codes only)."""
        col = df[name]
        if isinstance(col.dtype, pd.ArrowDtype) and str(col.dtype).startswith(
                "dictionary"):
            return pd.Categorical.from_codes(
                col._values.codes.astype("int64"),
                col._values.dictionary.to_pylist(),
                ordered=False)
        return col.astype("category")

    # 1. Low-cardinality text dimensions -> pandas category.
    for c in ["grade", "sub_grade", "home_ownership", "verification_status",
              "loan_status", "purpose", "addr_state", "application_type"]:
        df[c] = cat(c)

    # 2. Dates (issue_d/last_pymnt_d are dictionary-encoded "Jan-2015" labels;
    #    we decode them once into a datetime64 numpy array, vectorised).
    import pyarrow as pa
    import pyarrow.compute as pc
    for src, out in [("issue_d", "issue_date"), ("last_pymnt_d", "last_pymnt_date")]:
        d = df[src]
        if isinstance(d.dtype, pd.ArrowDtype) and str(d.dtype).startswith("dictionary"):
            codes = d._values.codes.to_numpy(zero_copy_only=False).astype("int64")
            dict_vals = d._values.dictionary.to_pylist()
            ts_map = []
            for v in dict_vals:
                try:
                    ts_map.append(pd.to_datetime(str(v), format="%b-%Y"))
                except (ValueError, TypeError):
                    ts_map.append(pd.NaT)
            base = np.array([t.value if t is not pd.NaT else -1 for t in ts_map],
                            dtype="int64")
            vals = np.where((codes >= 0) & (codes < len(base)), base[codes], -1)
            arr = vals.astype("datetime64[ns]")
            arr[arr.astype("int64") == -1] = np.datetime64("NaT")
            df[out] = arr
        else:
            df[out] = pd.to_datetime(d.astype("string"), format="%b-%Y",
                                     errors="coerce").to_numpy()

    # 3. Numeric typing (float32 already applied at read time; fill NaN->0
    #    where the business meaning of blank is 'no money').
    for c in ["total_rec_prncp", "recoveries", "out_prncp", "total_pymnt",
              "total_rec_int", "total_rec_late_fee"]:
        df[c] = df[c].astype("float32")
    df["delinq_2yrs"] = df["delinq_2yrs"].astype("Int16")
    df["open_acc"] = df["open_acc"].astype("Int16")

    # term months from dictionary like {" 36 months": ..., " 60 months": ...}
    t = df["term"]
    if isinstance(t.dtype, pd.ArrowDtype) and str(t.dtype).startswith("dictionary"):
        dict_vals = t._values.dictionary.to_pylist()
        months = [int(str(v).split()[0]) if isinstance(v, str) and str(v).strip()[:2].isdigit()
                  else -1 for v in dict_vals]
        df["term_months"] = np.asarray(months, dtype="int16")[
            t._values.codes.astype("int64")]
    else:
        df["term_months"] = (t.astype("string").str.extract(r"(\d+)")[0]
                             .astype("int16"))

    # FICO midpoint, computed block-wise in float32 (no big temporaries).
    lo = df["fico_range_low"].astype("float32").fillna(0).to_numpy(na_value=0)
    hi = df["fico_range_high"].astype("float32").fillna(0).to_numpy(na_value=0)
    df["fico"] = ((lo + hi) / 2).astype("float32")
    del lo, hi

    # 4. Derived DIMENSIONS only (bands/buckets are labels, not metrics).
    df["vintage"] = (df["issue_date"].astype("datetime64[ns]")
                     .dt.to_period("Q").astype(str).astype("category"))
    df["fico_band"] = pd.cut(df["fico"].astype("float32"),
                             [-np.inf, 659.999, 699.999, 739.999, np.inf],
                             labels=["<660", "660-699", "700-739", "740+"])
    df["dti_band"] = pd.cut(df["dti"].astype("float32"),
                            [-np.inf, 9.999, 19.999, 29.999, np.inf],
                            labels=["<10", "10-19", "20-29", "30+"])

    # 5. Flags (identical logic to the Tableau calcs in docs/calculations.md).
    df["default_flag"] = df["loan_status"].isin(DEFAULT_STATUSES).astype("int8")
    maturity = (pd.Series(df["issue_date"]).astype("datetime64[ns]")
                + pd.to_timedelta(df["term_months"].astype("int64"), unit="M"))
    df["mature_flag"] = (maturity <= MATURITY_CUTOFF).to_numpy().astype("int8")
    del maturity
    df["live_flag"] = df["loan_status"].isin(LIVE_STATUSES).astype("int8")

    live = df["live_flag"].to_numpy() == 1
    outp = df["out_prncp"].astype("float32").fillna(0).to_numpy()
    le = np.where(live, outp, 0.0).astype("float32")
    df["live_exposure"] = le
    del live, outp, le

    # 6. Net loss $ (row-level actual loss; the same formula lives in Tableau).
    funded = df["funded_amnt"].astype("float32").to_numpy()
    repaid = df["total_rec_prncp"].astype("float32").fillna(0).to_numpy()
    recov = df["recoveries"].astype("float32").fillna(0).to_numpy()
    loss = funded - repaid - recov
    df["net_loss"] = np.where((df["default_flag"].to_numpy() == 1) & (loss > 0),
                              loss, 0.0).astype("float32")
    del funded, repaid, recov, loss

    # 7. Delinquency bucket (dimension for the watchlist).
    df["delinq_bucket"] = df["loan_status"].map(
        {"In Grace Period": "Grace", "Late (16-30 days)": "16-30",
         "Late (31-120 days)": "31-120"}).fillna("").astype("category")

    df = df.drop(columns=["term", "issue_d", "last_pymnt_d",
                          "fico_range_low", "fico_range_high"])
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------
# Data-quality summary (second Tableau source -> Dashboard 6)
# --------------------------------------------------------------------------
def build_dq(df: pd.DataFrame) -> pd.DataFrame:
    outliers = {
        "dti": int(((df["dti"] < 0) | (df["dti"] > 60)).sum()),
        "annual_inc": int((df["annual_inc"] <= 0).sum()),
        "revol_util": int((df["revol_util"] > 100).sum()),
        "funded_amnt": int((df["funded_amnt"] <= 0).sum()),
        "int_rate": int((df["int_rate"] <= 0).sum()),
    }
    dq = pd.DataFrame({
        "field": [str(c) for c in df.columns],
        "data_type": df.dtypes.astype(str).values,
        "null_count": df.isna().sum().values,
        "null_pct": (df.isna().mean() * 100).round(3).values,
    })
    dq["outlier_count"] = dq["field"].map(outliers).fillna(0).astype(int)
    dq["outlier_rule"] = dq["field"].map({
        "dti": "outside 0-60",
        "annual_inc": "<= 0",
        "revol_util": "> 100",
        "funded_amnt": "<= 0",
        "int_rate": "<= 0",
    }).fillna("-")
    return dq


# Plain-English dictionary shown on Dashboard 6 and in the README.
DICTIONARY = [
    ("id", "Loan ID", "Unique number given to each loan when it was listed."),
    ("funded_amnt", "Funded Amount ($)", "Money actually lent to the borrower. The portfolio 'size' measure."),
    ("int_rate", "Interest Rate (%)", "The rate the borrower pays. Our revenue per loan."),
    ("grade / sub_grade", "Credit Grade (A-G)", "Lender's own risk rating, A = safest, G = riskiest."),
    ("fico", "Credit Score (avg)", "Borrower's external credit score (average of the reported range)."),
    ("dti", "Debt-to-Income", "Existing debt payments as a % of income. Higher = more stretched."),
    ("annual_inc", "Annual Income ($)", "Income declared by the borrower."),
    ("issue_date", "Issue Date", "When the loan was originated. Drives 'vintage' cohorts."),
    ("vintage", "Vintage", "The quarter a loan book was originated, e.g. 2015Q4. Loans from the same quarter behave together."),
    ("loan_status", "Loan Status", "Where the loan is today: paid, current, late, charged off..."),
    ("default_flag", "Default Flag (1/0)", "1 if the loan ended in loss (charged off / defaulted)."),
    ("mature_flag", "Mature Flag (1/0)", "1 if the loan's full term has run out by 2018-12-31. Only mature loans can be judged fairly."),
    ("live_flag", "Live Flag (1/0)", "1 if the loan is still open (current or late)."),
    ("out_prncp", "Outstanding Principal ($)", "Balance still owed today."),
    ("live_exposure", "Live Exposure ($)", "Outstanding principal, but only for loans still open. What is actually at risk."),
    ("total_rec_prncp", "Principal Repaid ($)", "Cash back from the borrower that reduced the balance."),
    ("recoveries", "Recoveries ($)", "Money clawed back after charge-off (collections, sale of debt)."),
    ("net_loss", "Net Loss ($)", "For defaulted loans: funded amount minus everything recovered. The real money lost."),
    ("purpose", "Purpose", "Why the borrower needed the money (debt consolidation, credit cards...)."),
    ("addr_state", "State", "Borrower's state of residence. Used for concentration analysis."),
    ("delinq_2yrs", "Past Delinquencies", "Times the borrower was 30+ days late in the 2 years before applying."),
    ("revol_util", "Revolving Utilisation (%)", "How much of existing credit-card limits was already used."),
    ("verification_status", "Income Verification", "Whether income was independently verified or self-reported."),
    ("fico_band / dti_band", "Risk Bands", "Groupings of scores / DTI into readable buckets (<660, 660-699, ...)."),
]


# --------------------------------------------------------------------------
# Validation: recompute the headline KPIs in python (must match Tableau)
# --------------------------------------------------------------------------
def validate(df: pd.DataFrame) -> dict:
    m = df[df.mature_flag == 1]
    default_rate_mature = m.default_flag.sum() / len(m)

    gcount = m.groupby("grade", observed=True)["id"].count()
    gdefs = m.groupby("grade", observed=True)["default_flag"].sum()
    pd_by_grade = (gdefs.astype("float64") / gcount.astype("float64"))

    defaults = df[df.default_flag == 1]
    lgd_den = float(defaults["net_loss"].astype("float64").sum())
    lgd = 1 - float(defaults["recoveries"].astype("float64").fillna(0).sum()) / lgd_den \
        if lgd_den else float("nan")

    live = df[df.live_flag == 1]
    row_el = (live["live_exposure"].astype("float64")
              * live["grade"].map(pd_by_grade).astype("float64") * lgd)
    el_dollars = float(row_el.sum())
    el_rate = el_dollars / float(live["live_exposure"].sum())

    coupon = float(np.average(df["int_rate"].astype("float64"),
                              weights=df["funded_amnt"].astype("float64")) / 100)
    exposure_years = (m.funded_amnt.astype("float64")
                      * m.term_months.astype("float64") / 12).sum()
    ann_loss = float(m.net_loss.sum() / exposure_years)
    cushion = coupon - ann_loss

    pdg_live = live["grade"].map(pd_by_grade).astype("float64")
    stressed = (live["live_exposure"].astype("float64")
                * (pdg_live * 2.0).clip(upper=1.0)
                * min(1.0, lgd + 0.10))
    stressed_el = float(stressed.sum())

    state_share = df.groupby("addr_state", observed=True).funded_amnt.sum() / df.funded_amnt.sum()
    hhi = float((state_share ** 2).sum() * 10000)
    top5 = float(state_share.sort_values(ascending=False).head(5).sum())

    grade_cushion = {}
    for g, gg in m.groupby("grade", observed=True):
        cp = (gg.int_rate.astype("float64") * gg.funded_amnt.astype("float64")).sum() / gg.funded_amnt.sum() / 100
        al = float(gg.net_loss.sum()) / float((gg.funded_amnt.astype("float64") * gg.term_months.astype("float64") / 12).sum())
        grade_cushion[g] = round(cp - al, 4)

    return {
        "rows": int(len(df)),
        "funded_amnt_total": float(df.funded_amnt.sum()),
        "default_rate_mature": round(float(default_rate_mature), 4),
        "pd_by_grade_mature": {k: round(float(v), 4) for k, v in pd_by_grade.items()},
        "lgd_portfolio": round(float(lgd), 4),
        "expected_loss_usd": round(el_dollars, 2),
        "el_rate": round(float(el_rate), 4),
        "weighted_coupon": round(coupon, 4),
        "annualised_loss_rate_mature": round(ann_loss, 4),
        "pricing_cushion": round(cushion, 4),
        "stressed_el_usd_pd2x_lgd_plus10": round(stressed_el, 2),
        "hhi_states": round(hhi, 1),
        "top5_state_share": round(top5, 4),
        "pricing_cushion_by_grade": grade_cushion,
    }


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", help=f"path to {RAW_CSV_NAME} (skips kagglehub download)")
    args = ap.parse_args()

    raw = resolve_raw_csv(args.csv)
    print(f"[prep] streaming {raw} (~1.6 GB, please wait)...")
    # STREAMING read: pyarrow opens the CSV as a RecordBatchReader and we
    # process one block at a time, writing compact Arrow IPC shards to disk.
    # Peak RAM stays low even though the file is 1.6 GB (a full in-memory
    # read would OOM an 8 GB laptop). Column pruning + float32 typing happen
    # inside the reader; low-cardinality text columns are dictionary-encoded.
    import pyarrow as pa
    import pyarrow.csv as pacsv
    import pyarrow.compute as pc
    import pyarrow.ipc as ipc

    read_opts = pacsv.ReadOptions(block_size=16 * 1024 * 1024,
                                  autogenerate_column_names=False)
    conv_opts = pacsv.ConvertOptions(
        include_columns=COLS,
        strings_can_be_null=True,
        null_values=["", "nan", "NaN", "NULL", "null"],
        auto_dict_encode=True,
        auto_dict_max_cardinality=512,
        column_types={c: pa.float32() for c in [
            "loan_amnt", "funded_amnt", "int_rate", "installment", "annual_inc",
            "dti", "revol_util", "out_prncp", "total_pymnt", "total_rec_prncp",
            "total_rec_int", "total_rec_late_fee", "recoveries"]},
    )
    parse_opts = pacsv.ParseOptions(delimiter=",", quoting=False)

    shard_dir = OUT_DIR / "_shards"
    shard_dir.mkdir(exist_ok=True)
    for old in shard_dir.glob("*.arrow"):
        old.unlink()

    n_rows = 0
    n_shard = 0
    with pacsv.open_csv(str(raw), read_options=read_opts,
                        convert_options=conv_opts,
                        parse_options=parse_opts) as reader:
        missing = set(COLS) - set(reader.schema.names)
        if missing:
            sys.exit(f"[prep] columns missing from raw file: {missing}")
        id_field = reader.schema.names.index("id")

        buf: list[pa.Table] = []
        buf_rows = 0

        def flush() -> None:
            nonlocal buf, buf_rows, n_shard
            if not buf:
                return
            big = pa.concat_tables(buf)
            with ipc.new_file(str(shard_dir / f"s{n_shard}.arrow"), big.schema) as w:
                for b in big.to_batches():
                    w.write_batch(b)
            del big
            buf, buf_rows, n_shard = [], 0, n_shard + 1
            gc.collect()

        for batch in reader:
            ids = batch.column(id_field)
            keep = pc.and_(pc.is_valid(ids), pc.match_regex(ids, r"^[0-9]+$"))
            n_keep = pc.sum(keep.cast(pa.int64())).as_py() or 0
            if n_keep == 0:
                continue
            tbl = pa.Table.from_batches([batch]).filter(keep)
            buf.append(tbl)
            buf_rows += n_keep
            n_rows += n_keep
            if buf_rows >= 300_000:
                flush()
        flush()

    print(f"[prep] {n_rows:,} valid loan rows in {n_shard} arrow shards")

    # Reassemble shards into one pandas frame (ArrowDtype, dict-encoded).
    tables = [ipc.open_file(p).read() for p in sorted(shard_dir.glob("*.arrow"))]
    big = pa.concat_tables(tables)
    del tables
    gc.collect()
    df = big.to_pandas(types_mapper=pd.ArrowDtype)
    del big
    gc.collect()
    df["id"] = pd.to_numeric(df["id"].astype("string"), errors="coerce").astype("int64")
    for old in shard_dir.glob("*.arrow"):
        old.unlink()
    print(f"[prep] loaded {len(df):,} loan rows")

    df = clean(df)
    gc.collect()

    dq = build_dq(df)
    val = validate(df)

    pd.DataFrame({"field": [d[0] for d in DICTIONARY],
                  "plain_english_name": [d[1] for d in DICTIONARY],
                  "what_it_means": [d[2] for d in DICTIONARY]}
                 ).to_csv(OUT_DIR / "dictionary.csv", index=False)

    smap = (df.groupby("loan_status", observed=True).size().rename("loans").reset_index())
    smap["risk_bucket"] = smap["loan_status"].map(STATUS_BUCKET).fillna("Other")
    smap["is_default_definition"] = smap["loan_status"].isin(DEFAULT_STATUSES)
    smap["is_live_definition"] = smap["loan_status"].isin(LIVE_STATUSES)
    smap["share_pct"] = (smap["loans"] / smap["loans"].sum() * 100).round(2)
    smap.to_csv(OUT_DIR / "status_map.csv", index=False)

    dq.to_csv(OUT_DIR / "dq_summary.csv", index=False)
    df.to_csv(OUT_DIR / "loans_clean.csv", index=False)

    with open(OUT_DIR / "reconciliation.txt", "w") as fh:
        fh.write("RECONCILIATION CONTROL (source vs Tableau must match)\n")
        fh.write("=" * 52 + "\n")
        fh.write(f"rows_in_loans_clean.csv : {val['rows']:,}\n")
        fh.write(f"SUM(funded_amnt)        : ${val['funded_amnt_total']:,.2f}\n")
        fh.write(f"SUM(live_exposure)      : ${df.live_exposure.sum():,.2f}\n")
        fh.write(f"SUM(net_loss)           : ${df.net_loss.sum():,.2f}\n")

    with open(OUT_DIR / "calc_validation.json", "w") as fh:
        json.dump(val, fh, indent=2)

    print("\n[prep] done. Files in", OUT_DIR)
    print(json.dumps(val, indent=2)[:1500])


if __name__ == "__main__":
    main()
