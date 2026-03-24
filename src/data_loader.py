"""
Data loading and parsing utilities for the Synergy Calculation Tool.

Excel structure (confirmed from sample workbook):
  Weekly              : Col0=ModelKey, Col1=Variable(clean), Cols2-9=metadata/FY-agg, Cols10+=dates
  WeeklyTransformSupport: Col0=Model,  Col1=Variables(clean), Cols2+=dates
  WeeklySupport       : Col0=Variable, Cols1-3=metadata(Factor,UnitName,AggRule), Cols4+=dates
  WeeklySpend         : Col0=Variable, Cols1+=dates
  Model detail sheets : Row where Col0=='Type' is the header:
                        Col0=Type, Col1=Bucket, Col2=Description,
                        Col3=Variable(clean), Col4=Transformation, Col5=PostMultiplier
  Transformation format: VARNAME__APL_adstock|power|lag  or  VARNAME__RA_weeks
  Date format         : MM/DD/YYYY strings
"""

import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import openpyxl
import pandas as pd

BASE_PATH = Path("core_workbook")


# ---------------------------------------------------------------------------
# Country / folder helpers
# ---------------------------------------------------------------------------

def get_countries() -> list:
    if not BASE_PATH.exists():
        return []
    return sorted(f.name for f in BASE_PATH.iterdir() if f.is_dir())


def validate_country_folder(country: str) -> Tuple[bool, str, Optional[Path]]:
    folder = BASE_PATH / country
    xlsx_files = list(folder.glob("*.xlsx")) + list(folder.glob("*.xlsm"))
    if len(xlsx_files) == 0:
        return False, f"No Excel file found in the '{country}' folder.", None
    if len(xlsx_files) > 1:
        return (
            False,
            "Place only the latest version of the Core workbook in the relevant "
            "folder, do not have multiple files",
            None,
        )
    return True, "", xlsx_files[0]


# ---------------------------------------------------------------------------
# Date detection
# ---------------------------------------------------------------------------

def _try_parse_date(val) -> Optional[pd.Timestamp]:
    """Return a Timestamp if val looks like MM/DD/YYYY or DD/MM/YYYY, else None."""
    if val is None:
        return None
    s = str(val).strip()
    for fmt in ("%m/%d/%Y", "%d/%m/%Y"):
        try:
            return pd.to_datetime(s, format=fmt)
        except Exception:
            pass
    return None


def _parse_cell_date(val) -> Optional[pd.Timestamp]:
    """Parse a cell value that may be a Python datetime/date object or a string."""
    if val is None:
        return None
    if hasattr(val, "year"):          # datetime / date from openpyxl
        return pd.Timestamp(val)
    return _try_parse_date(val)


def _date_cols(columns) -> list:
    return [c for c in columns if _try_parse_date(c) is not None]


# ---------------------------------------------------------------------------
# Reporting periods (Summary sheet)
# ---------------------------------------------------------------------------

def _parse_reporting_periods(rows: list) -> list:
    """
    Find the reporting-periods table in the Summary sheet.

    Scans for a header row that contains all three of:
      'Reporting Period', 'Start Date', 'End Date'
    (case-insensitive, any column order).

    Returns a list of dicts: [{period, start, end}, ...]
    where start/end are pd.Timestamp.
    """
    if not rows:
        return []

    header_idx = col_period = col_start = col_end = None
    for i, row in enumerate(rows):
        if not row:
            continue
        lower = [str(c).strip().lower() if c is not None else "" for c in row]
        if (
            "reporting period" in lower
            and "start date" in lower
            and "end date" in lower
        ):
            header_idx = i
            col_period = lower.index("reporting period")
            col_start  = lower.index("start date")
            col_end    = lower.index("end date")
            break

    if header_idx is None:
        return []

    periods = []
    for row in rows[header_idx + 1:]:
        if not row:
            continue
        max_col = max(col_period, col_start, col_end)
        if len(row) <= max_col:
            continue
        period_val = row[col_period]
        start_val  = row[col_start]
        end_val    = row[col_end]
        if not period_val or str(period_val).strip() in ("", "None", "nan"):
            break   # treat first empty row as end of table
        start_dt = _parse_cell_date(start_val)
        end_dt   = _parse_cell_date(end_val)
        if start_dt is not None and end_dt is not None:
            periods.append({
                "period": str(period_val).strip(),
                "start":  start_dt,
                "end":    end_dt,
            })

    return periods


# ---------------------------------------------------------------------------
# Raw sheet loading (all sheets, including hidden)
# ---------------------------------------------------------------------------

def _load_raw_sheets(xlsx_path: Path) -> dict:
    """Return {sheet_title: list_of_row_tuples} for every sheet (incl. hidden)."""
    wb = openpyxl.load_workbook(xlsx_path, data_only=True)
    sheets = {ws.title: list(ws.values) for ws in wb.worksheets}
    wb.close()
    return sheets


# ---------------------------------------------------------------------------
# Sheet parsers
# ---------------------------------------------------------------------------

def _parse_contribution_sheet(rows: list) -> pd.DataFrame:
    """
    Parse Weekly or WeeklyTransformSupport rows.
    Uses row[0] as headers; keeps col[0] (model), col[1] (variable), + date cols.
    """
    if not rows:
        return pd.DataFrame()
    headers = rows[0]
    dc = _date_cols(headers)
    if not dc:
        return pd.DataFrame()

    keep = [headers[0], headers[1]] + dc
    data = []
    for row in rows[1:]:
        if len(row) < 2:
            continue
        model = row[0]
        variable = row[1]
        if not model or not variable:
            continue
        row_dict = {"model": str(model).strip(), "variable": str(variable).strip()}
        for col in dc:
            idx = list(headers).index(col)
            row_dict[col] = pd.to_numeric(row[idx] if idx < len(row) else None, errors="coerce")
        data.append(row_dict)

    return pd.DataFrame(data)


def _parse_support_sheet(rows: list) -> pd.DataFrame:
    """
    Parse WeeklySupport or WeeklySpend rows.
    Col0=Variable, rest detected by date parsing (skips Factor/UnitName/AggRule etc.).
    """
    if not rows:
        return pd.DataFrame()
    headers = rows[0]
    dc = _date_cols(headers)
    if not dc:
        return pd.DataFrame()

    data = []
    for row in rows[1:]:
        if not row or not row[0]:
            continue
        row_dict = {"variable": str(row[0]).strip()}
        for col in dc:
            idx = list(headers).index(col)
            row_dict[col] = pd.to_numeric(row[idx] if idx < len(row) else None, errors="coerce")
        data.append(row_dict)

    return pd.DataFrame(data)


def _parse_model_detail_sheet(rows: list) -> pd.DataFrame:
    """
    Parse a model detail sheet.
    Finds the row where col[0]=='Type' as the true header, then reads:
      col[1]=Bucket, col[2]=Description, col[3]=Variable(clean), col[4]=Transformation
    Returns DataFrame with columns: variable, bucket, description, transformation.
    """
    header_idx = None
    for i, row in enumerate(rows):
        if row and str(row[0]).strip() == "Type":
            header_idx = i
            break

    if header_idx is None:
        return pd.DataFrame(columns=["variable", "bucket", "description", "transformation"])

    records = []
    for row in rows[header_idx + 1:]:
        if not row or len(row) < 4:
            continue
        var = row[3]
        if not var or str(var).strip().lower() in ("", "none", "nan"):
            continue
        records.append({
            "variable":       str(var).strip(),
            "bucket":         str(row[1]).strip() if row[1] else "",
            "description":    str(row[2]).strip() if row[2] else "",
            "transformation": str(row[4]).strip() if len(row) > 4 and row[4] else "",
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Transformation string parser
# ---------------------------------------------------------------------------

def parse_transformation(transformation: str) -> dict:
    """
    Parse a transformation string like:
      VARNAME__APL_0.65|0.6|0   → adstock=0.65, power=0.6, lag=0
      VARNAME__RA_7              → rolling_avg=7
      VARNAME                    → no transformation
    Returns dict with keys: adstock, power, lag, rolling_avg (None if not present).
    """
    result = {"adstock": None, "power": None, "lag": None, "rolling_avg": None}
    if not transformation:
        return result

    # Split on double-underscore to separate var name from transform params
    parts = transformation.split("__", 1)
    if len(parts) < 2:
        return result

    params = parts[1]

    # APL_adstock|power|lag
    m_apl = re.match(r"APL_([\d.]+)\|([\d.]+)\|([\d.]+)", params)
    if m_apl:
        result["adstock"] = float(m_apl.group(1))
        result["power"] = float(m_apl.group(2))
        result["lag"] = int(float(m_apl.group(3)))
        return result

    # RA_weeks
    m_ra = re.match(r"RA_(\d+)", params)
    if m_ra:
        result["rolling_avg"] = int(m_ra.group(1))

    return result


# ---------------------------------------------------------------------------
# Time series extractor
# ---------------------------------------------------------------------------

def get_series(df: pd.DataFrame, model: str, variable: str) -> pd.Series:
    """
    Extract a time series from a contribution-style DataFrame
    (Weekly or WeeklyTransformSupport).
    Returns a DatetimeIndex-indexed Series with NaN rows dropped.
    """
    mask = (df["model"].str.strip() == model.strip()) & (
        df["variable"].str.strip() == variable.strip()
    )
    rows = df[mask]
    if rows.empty:
        return pd.Series(dtype=float)

    date_cols = [c for c in df.columns if c not in ("model", "variable")]
    row = rows.iloc[0]
    vals = pd.to_numeric(row[date_cols], errors="coerce").values

    dates = [_try_parse_date(c) for c in date_cols]
    series = pd.Series(vals, index=pd.DatetimeIndex(dates))
    return series.dropna()


# ---------------------------------------------------------------------------
# Model-dependent variable mapping (Summary sheet — Model Details table)
# ---------------------------------------------------------------------------

def _parse_model_dependents(rows: list) -> dict:
    """
    Find the model/dependent table in the Summary sheet.

    Scans for a header row containing both 'model' and 'dependent'
    (case-insensitive).  'Model Details' is treated as an optional label
    column and is ignored.

    Returns {model_name: dependent_variable_name}.
    """
    if not rows:
        return {}

    header_idx = col_model = col_dep = None
    for i, row in enumerate(rows):
        if not row:
            continue
        lower = [str(c).strip().lower() if c is not None else "" for c in row]
        # must have exact cells "model" and "dependent"
        if "model" in lower and "dependent" in lower:
            header_idx = i
            col_model  = lower.index("model")
            col_dep    = lower.index("dependent")
            break

    if header_idx is None:
        return {}

    result = {}
    for row in rows[header_idx + 1:]:
        if not row:
            continue
        max_col = max(col_model, col_dep)
        if len(row) <= max_col:
            continue
        model_val = row[col_model]
        dep_val   = row[col_dep]
        if not model_val or str(model_val).strip() in ("", "None", "nan"):
            break   # first empty row ends the table
        if dep_val and str(dep_val).strip() not in ("", "None", "nan"):
            result[str(model_val).strip()] = str(dep_val).strip()

    return result


# ---------------------------------------------------------------------------
# Total model contributions
# ---------------------------------------------------------------------------

# Variables to exclude from the total — these are derived/predicted rows, not real drivers
_EXCLUDE_VARS = {"Predicted", "Intercept", "Base", "Residual"}

def get_total_model_contributions(
    weekly_df: pd.DataFrame,
    model: str,
    dependent_var: Optional[str] = None,
) -> pd.Series:
    """
    Sum all driver contributions for a model across all variables (by week).

    Excludes meta-rows like 'Predicted', 'Intercept', 'Base', 'Residual'
    and, when supplied, the model's own dependent variable (to prevent
    double-counting: sum(predictors) == dependent, so including the dependent
    row would inflate the total by ~2×).

    Returns a DatetimeIndex-indexed Series.
    """
    exclude = _EXCLUDE_VARS
    if dependent_var:
        exclude = exclude | {dependent_var}

    mask = weekly_df["model"].str.strip() == model.strip()
    rows = weekly_df[mask]
    rows = rows[~rows["variable"].str.strip().isin(exclude)]

    if rows.empty:
        return pd.Series(dtype=float)

    date_cols = [c for c in weekly_df.columns if c not in ("model", "variable")]

    totals = {}
    for col in date_cols:
        dt = _try_parse_date(col)
        if dt is None:
            continue
        vals = pd.to_numeric(rows[col], errors="coerce").fillna(0)
        totals[dt] = float(vals.sum())

    if not totals:
        return pd.Series(dtype=float)

    s = pd.Series(totals).sort_index()
    return s


# ---------------------------------------------------------------------------
# Top-level loader
# ---------------------------------------------------------------------------

def load_country_data(country: str) -> Tuple[bool, str, dict]:
    """
    Load all data for a country.
    Returns (success, error_message, data_dict).

    data_dict keys:
      weekly, weekly_transform_support, weekly_support, weekly_spend  → DataFrames
      model_details  → {model_name: DataFrame}
      variable_meta  → DataFrame(model, variable, bucket, description, transformation)
    """
    valid, error, xlsx_path = validate_country_folder(country)
    if not valid:
        return False, error, {}

    raw = _load_raw_sheets(xlsx_path)
    data: dict = {}

    # ---- contribution sheets ----
    for key, sheet_name in [
        ("weekly", "Weekly"),
        ("weekly_transform_support", "WeeklyTransformSupport"),
    ]:
        data[key] = _parse_contribution_sheet(raw.get(sheet_name, []))

    # ---- support/spend sheets ----
    for key, sheet_name in [
        ("weekly_support", "WeeklySupport"),
        ("weekly_spend", "WeeklySpend"),
    ]:
        data[key] = _parse_support_sheet(raw.get(sheet_name, []))

    # ---- model detail sheets ----
    weekly_df = data["weekly"]
    models = (
        weekly_df["model"].dropna().unique().tolist()
        if not weekly_df.empty and "model" in weekly_df.columns
        else []
    )

    model_details: dict = {}
    meta_rows: list = []

    for model in models:
        model_str = str(model).strip()
        if model_str in raw:
            detail_df = _parse_model_detail_sheet(raw[model_str])
            model_details[model_str] = detail_df
            if not detail_df.empty:
                detail_df = detail_df.copy()
                detail_df.insert(0, "model", model_str)
                meta_rows.append(detail_df)

    data["model_details"] = model_details

    # ---- Summary sheet: reporting periods + model dependents ----
    summary_rows = raw.get("Summary", [])
    data["reporting_periods"] = _parse_reporting_periods(summary_rows)
    data["model_dependents"]  = _parse_model_dependents(summary_rows)

    # ---- consolidated variable metadata ----
    if meta_rows:
        data["variable_meta"] = pd.concat(meta_rows, ignore_index=True)
    else:
        data["variable_meta"] = pd.DataFrame(
            columns=["model", "variable", "bucket", "description", "transformation"]
        )

    return True, "", data
