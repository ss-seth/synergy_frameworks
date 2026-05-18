"""
Synergy Calculation Tool — Streamlit app.

Workflow:
  1. Download the pre-filled selection template, mark 'y' in 'Include in Test'
  2. Upload the completed template — pairs are previewed grouped by model
  3. Configure CI + bootstrap settings
  4. Run analysis → one NNLS model per unique variable pair
  5. View significant synergies + download ZIP with Excel/PDF per model
"""

import io
import re
import zipfile
from collections import defaultdict
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import xlsxwriter as _xlsxwriter

from src.data_loader import (
    get_countries,
    get_series,
    get_total_model_contributions,
    load_country_data,
)
from src.output_export import create_synergy_chart, export_to_excel, export_to_pdf
from src.synergy_model import compute_synergy_model

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Synergy Calculator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    "<style>[data-testid='stAppDeployButton'] {display: none;}</style>",
    unsafe_allow_html=True,
)

# ── Sidebar: settings ─────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Synergy Calculator")
    st.divider()

    countries = get_countries()
    if not countries:
        st.error("No country folders found in `core_workbook/`.")
        st.stop()

    selected_country = st.selectbox("Country", countries)

    st.divider()
    st.subheader("Model Settings")

    ci_level = st.selectbox(
        "Confidence Interval",
        options=[0.80, 0.90, 0.95, 0.99],
        index=2,
        format_func=lambda x: f"{int(x * 100)}%",
        help="Bootstrap percentile CI level applied to all coefficient estimates.",
    )

    n_bootstrap = st.select_slider(
        "Bootstrap Iterations",
        options=[500, 1000, 2000, 5000],
        value=1000,
        help="Higher = more accurate CIs but slower.",
    )

    st.divider()
    st.caption("Estimation: NNLS — no intercept — no seasonality")
    st.caption("Only pairs with a statistically significant positive synergy coefficient are reported.")
    st.caption("Requires: R² ≥ 0, p-value < 0.05, bootstrap CI lower > 0, ΔR² > 0.001")


# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading workbook…")
def _load(country: str):
    return load_country_data(country)


success, err_msg, data = _load(selected_country)

if not success:
    st.error(err_msg)
    st.stop()

weekly           = data.get("weekly",                    pd.DataFrame())
wts              = data.get("weekly_transform_support",  pd.DataFrame())
var_meta         = data.get("variable_meta",             pd.DataFrame())
model_dependents = data.get("model_dependents",          {})

if weekly.empty:
    st.error("Sheet 'Weekly' not found or empty.")
    st.stop()
if wts.empty:
    st.error("Sheet 'WeeklyTransformSupport' not found or empty.")
    st.stop()

# ── Reporting period selector (sidebar, post-load) ────────────────────────────
reporting_periods    = data.get("reporting_periods", [])
period_start         = None
period_end           = None
selected_period_name = "Full range"

with st.sidebar:
    st.divider()
    st.subheader("Reporting Period")
    if reporting_periods:
        period_opts          = ["Full range"] + [p["period"] for p in reporting_periods]
        selected_period_name = st.selectbox("Select period", period_opts, key="period_sel")
        if selected_period_name != "Full range":
            sel_p        = next(p for p in reporting_periods if p["period"] == selected_period_name)
            period_start = sel_p["start"]
            period_end   = sel_p["end"]
            st.caption(f"{period_start.strftime('%d %b %Y')}  –  {period_end.strftime('%d %b %Y')}")
    else:
        st.caption("No reporting periods found in Summary sheet — using full date range.")

    st.divider()
    st.subheader("Output Folder")
    _default_folder = str(Path.home() / "Desktop" / "synergy_output")
    _sidebar_out_folder = st.text_input(
        "Save results to",
        value=st.session_state.get("_out_folder", _default_folder),
        help="Significant synergies are written here automatically as each model finishes. Leave blank to disable.",
    )
    st.session_state["_out_folder"] = _sidebar_out_folder
    if _sidebar_out_folder:
        st.caption("Files saved automatically per model during analysis.")
    else:
        st.caption("Leave blank to skip auto-save.")


def _clip(s: pd.Series, start, end) -> pd.Series:
    if start is not None:
        s = s[s.index >= start]
    if end is not None:
        s = s[s.index <= end]
    return s


# ── Template generation ───────────────────────────────────────────────────────

def _build_classification_lookup(weekly_df: pd.DataFrame) -> dict:
    """Map (model, variable) -> classification string from the Weekly sheet."""
    lookup = {}
    if weekly_df.empty or "classification" not in weekly_df.columns:
        return lookup
    sub = weekly_df[["model", "variable", "classification"]].dropna(subset=["classification"])
    for _, row in sub.iterrows():
        cls = str(row["classification"]).strip()
        if cls and cls.lower() not in ("none", "nan", ""):
            lookup[(str(row["model"]).strip(), str(row["variable"]).strip())] = cls
    return lookup


@st.cache_data(show_spinner=False)
def generate_selection_template(country: str) -> bytes:
    """
    Generate a pre-filled Excel selection template for the given country.
    Returns raw bytes of the .xlsx file.
    """
    buf = io.BytesIO()
    wb = _xlsxwriter.Workbook(buf, {"in_memory": True})

    hdr_fmt  = wb.add_format({"bold": True, "bg_color": "#2E4057", "font_color": "white",
                               "border": 1, "align": "center", "valign": "vcenter"})
    body_fmt = wb.add_format({"border": 1, "valign": "vcenter"})
    note_fmt = wb.add_format({"bold": True, "font_size": 13})
    text_fmt = wb.add_format({"font_size": 10})

    class_lookup = _build_classification_lookup(weekly)

    # ── Instructions tab ──────────────────────────────────────────────────────
    ws_instr = wb.add_worksheet("Instructions")
    ws_instr.write(0, 0, "Synergy Calculation — Selection Template", note_fmt)
    lines = [
        "",
        "HOW TO USE THIS TEMPLATE:",
        "",
        "1. Cross-Bucket Synergies tab:",
        "   Each row represents one (Model, Bucket) combination.",
        "   Type 'y' in the 'Include in Test' column for every bucket you want to test.",
        "   All variables in selected buckets will be tested pairwise against variables in",
        "   OTHER selected buckets within the SAME model. Cross-model pairs are never tested.",
        "",
        "2. Intra-Bucket Synergies tab:",
        "   Each row represents one variable. Type 'y' in 'Include in Test' for variables",
        "   you want to include. Selected variables will be tested pairwise against other",
        "   selected variables in the SAME bucket and model.",
        "",
        "3. Save the file and upload it back into the app.",
        "",
        "NOTE: The 'Grouping' column (Base/Incremental) is a reference label only.",
        "      It does not control which pairs are tested.",
    ]
    for i, line in enumerate(lines):
        ws_instr.write(i + 1, 0, line, text_fmt)
    ws_instr.set_column(0, 0, 100)

    # ── Cross-Bucket Synergies tab ────────────────────────────────────────────
    ws_cross = wb.add_worksheet("Cross-Bucket Synergies")
    cross_headers = ["Model", "Bucket", "# Variables", "Grouping", "Include in Test"]
    col_widths_cross = [32, 28, 12, 16, 18]
    for c_idx, (h, w) in enumerate(zip(cross_headers, col_widths_cross)):
        ws_cross.write(0, c_idx, h, hdr_fmt)
        ws_cross.set_column(c_idx, c_idx, w)

    row_idx = 1
    if not var_meta.empty:
        for (model, bucket), grp in var_meta.groupby(["model", "bucket"], sort=True):
            model_s  = str(model).strip()
            bucket_s = str(bucket).strip()
            if not bucket_s or bucket_s.lower() in ("none", "nan", ""):
                continue

            var_list = grp["variable"].tolist()
            clsf = [class_lookup.get((model_s, v), "") for v in var_list]
            clsf = [c for c in clsf if c]
            if not clsf:
                grouping = ""
            elif len(set(clsf)) == 1:
                grouping = clsf[0]
            else:
                grouping = pd.Series(clsf).value_counts().index[0]

            ws_cross.write(row_idx, 0, model_s,         body_fmt)
            ws_cross.write(row_idx, 1, bucket_s,        body_fmt)
            ws_cross.write(row_idx, 2, len(grp),        body_fmt)
            ws_cross.write(row_idx, 3, grouping,        body_fmt)
            ws_cross.write(row_idx, 4, "",              body_fmt)
            row_idx += 1

    # ── Intra-Bucket Synergies tab ────────────────────────────────────────────
    ws_intra = wb.add_worksheet("Intra-Bucket Synergies")
    intra_headers = ["Model", "Bucket", "Variable", "Description", "Grouping", "Include in Test"]
    col_widths_intra = [32, 28, 28, 44, 16, 18]
    for c_idx, (h, w) in enumerate(zip(intra_headers, col_widths_intra)):
        ws_intra.write(0, c_idx, h, hdr_fmt)
        ws_intra.set_column(c_idx, c_idx, w)

    row_idx = 1
    if not var_meta.empty:
        for _, row in var_meta.sort_values(["model", "bucket", "variable"]).iterrows():
            model_s  = str(row["model"]).strip()
            bucket_s = str(row.get("bucket", "") or "").strip()
            var_s    = str(row["variable"]).strip()
            desc_s   = str(row.get("description", "") or "").strip()
            if not bucket_s or bucket_s.lower() in ("none", "nan", ""):
                continue
            grouping = class_lookup.get((model_s, var_s), "")
            ws_intra.write(row_idx, 0, model_s,  body_fmt)
            ws_intra.write(row_idx, 1, bucket_s, body_fmt)
            ws_intra.write(row_idx, 2, var_s,    body_fmt)
            ws_intra.write(row_idx, 3, desc_s,   body_fmt)
            ws_intra.write(row_idx, 4, grouping, body_fmt)
            ws_intra.write(row_idx, 5, "",       body_fmt)
            row_idx += 1

    wb.close()
    buf.seek(0)
    return buf.read()


# ── Template parsing ──────────────────────────────────────────────────────────

def parse_selection_template(uploaded_file) -> tuple:
    """
    Parse an uploaded selection template.

    Returns (cross_bucket_selections, intra_var_selections, error_str_or_None)
      cross_bucket_selections : list of (model, bucket)
      intra_var_selections    : dict of {(model, bucket): [var, ...]}
    """
    try:
        xl = pd.ExcelFile(uploaded_file, engine="openpyxl")
    except Exception as exc:
        return [], {}, f"Could not read uploaded file: {exc}"

    cross_selections = []
    intra_selections: dict = defaultdict(list)

    # Cross-Bucket tab
    if "Cross-Bucket Synergies" in xl.sheet_names:
        try:
            df = xl.parse("Cross-Bucket Synergies", dtype=str)
            df.columns = [str(c).strip() for c in df.columns]
            needed = {"Model", "Bucket", "Include in Test"}
            if needed.issubset(set(df.columns)):
                for _, row in df.iterrows():
                    val = str(row.get("Include in Test", "") or "").strip().lower()
                    if val == "y":
                        model  = str(row["Model"] or "").strip()
                        bucket = str(row["Bucket"] or "").strip()
                        if model and bucket and model.lower() not in ("nan", "none"):
                            cross_selections.append((model, bucket))
        except Exception as exc:
            return [], {}, f"Error reading 'Cross-Bucket Synergies' tab: {exc}"

    # Intra-Bucket tab
    if "Intra-Bucket Synergies" in xl.sheet_names:
        try:
            df = xl.parse("Intra-Bucket Synergies", dtype=str)
            df.columns = [str(c).strip() for c in df.columns]
            needed = {"Model", "Variable", "Include in Test"}
            if needed.issubset(set(df.columns)):
                for _, row in df.iterrows():
                    val = str(row.get("Include in Test", "") or "").strip().lower()
                    if val == "y":
                        model  = str(row["Model"]    or "").strip()
                        bucket = str(row.get("Bucket", "") or "").strip()
                        var    = str(row["Variable"] or "").strip()
                        if model and var and model.lower() not in ("nan", "none"):
                            intra_selections[(model, bucket)].append(var)
        except Exception as exc:
            return [], {}, f"Error reading 'Intra-Bucket Synergies' tab: {exc}"

    return cross_selections, dict(intra_selections), None


# ── Main UI ───────────────────────────────────────────────────────────────────
st.header(f"Synergy Analysis — {selected_country}")
st.divider()

# ── Step 1: Template download + upload ───────────────────────────────────────
st.subheader("1  Select Pairs via Template")
st.markdown(
    "Download the pre-filled template below, mark **y** in the *Include in Test* column "
    "for the buckets and variables you want, then upload the completed file."
)

dl_col, _, info_col = st.columns([2, 1, 3])
with dl_col:
    template_bytes = generate_selection_template(selected_country)
    now_str = datetime.now().strftime("%H%M")
    st.download_button(
        label="📥 Download Selection Template",
        data=template_bytes,
        file_name=f"synergy_template_{selected_country}_{now_str}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
        key="dl_template",
    )
with info_col:
    st.caption(
        "Tab 1 — **Cross-Bucket Synergies**: select buckets. "
        "Variables across different selected buckets (same model) will be tested. "
        "Tab 2 — **Intra-Bucket Synergies**: select individual variables within a bucket."
    )

uploaded_template = st.file_uploader(
    "Upload completed template",
    type=["xlsx"],
    key=f"template_upload_{selected_country}",
    help="Upload the filled-in template to define which pairs to test.",
)

# Parse on upload, cache result in session_state keyed by (name, size, country)
cross_bucket_selections: list = []
intra_var_selections: dict    = {}
template_error = None

if uploaded_template is not None:
    file_id = (uploaded_template.name, uploaded_template.size, selected_country)
    if st.session_state.get("_tpl_file_id") != file_id:
        c_sel, i_sel, t_err = parse_selection_template(uploaded_template)
        st.session_state["_tpl_file_id"]    = file_id
        st.session_state["_tpl_cross"]      = c_sel
        st.session_state["_tpl_intra"]      = i_sel
        st.session_state["_tpl_error"]      = t_err

    cross_bucket_selections = st.session_state.get("_tpl_cross", [])
    intra_var_selections    = st.session_state.get("_tpl_intra", {})
    template_error          = st.session_state.get("_tpl_error")

    if template_error:
        st.error(f"Template error: {template_error}")
    else:
        # ── Preview ──────────────────────────────────────────────────────────
        total_cross_pairs = 0
        total_intra_pairs = 0

        # Group cross-bucket by model
        model_to_cross_buckets: dict = defaultdict(list)
        for model, bucket in cross_bucket_selections:
            model_to_cross_buckets[model].append(bucket)

        # Intra by model
        model_to_intra: dict = defaultdict(dict)
        for (model, bucket), vars_list in intra_var_selections.items():
            model_to_intra[model][(model, bucket)] = vars_list

        all_preview_models = sorted(
            set(list(model_to_cross_buckets.keys()) + list(model_to_intra.keys()))
        )

        if not all_preview_models:
            st.warning("No pairs selected. Mark at least one bucket or variable with 'y' in the template.")
        else:
            st.markdown("**Preview — pairs to be tested:**")
            for model in all_preview_models:
                cross_buckets = model_to_cross_buckets.get(model, [])
                intra_entries = {
                    k: v for k, v in intra_var_selections.items() if k[0] == model
                }

                # Count cross pairs for this model
                model_cross_pairs = 0
                for b1, b2 in combinations(cross_buckets, 2):
                    if not var_meta.empty:
                        n1 = len(var_meta[(var_meta["model"] == model) & (var_meta["bucket"] == b1)])
                        n2 = len(var_meta[(var_meta["model"] == model) & (var_meta["bucket"] == b2)])
                        model_cross_pairs += n1 * n2

                # Count intra pairs for this model
                model_intra_pairs = sum(
                    len(vl) * (len(vl) - 1) // 2
                    for vl in intra_entries.values()
                    if len(vl) >= 2
                )

                total_cross_pairs += model_cross_pairs
                total_intra_pairs += model_intra_pairs

                parts = []
                if model_cross_pairs:
                    parts.append(f"{model_cross_pairs} cross-bucket pair{'s' if model_cross_pairs != 1 else ''}")
                if model_intra_pairs:
                    parts.append(f"{model_intra_pairs} intra-bucket pair{'s' if model_intra_pairs != 1 else ''}")
                summary_str = " + ".join(parts) if parts else "0 pairs"

                with st.expander(f"**{model}** — {summary_str}", expanded=False):
                    if cross_buckets:
                        st.markdown("*Cross-bucket buckets selected:*")
                        for b in sorted(cross_buckets):
                            if not var_meta.empty:
                                n = len(var_meta[(var_meta["model"] == model) & (var_meta["bucket"] == b)])
                            else:
                                n = 0
                            st.markdown(f"  - **{b}** ({n} variable{'s' if n != 1 else ''})")
                    if intra_entries:
                        st.markdown("*Intra-bucket variables selected:*")
                        for (m, bkt), vlist in intra_entries.items():
                            st.markdown(f"  - **{bkt}**: {', '.join(vlist)}")

            total_pairs = total_cross_pairs + total_intra_pairs
            if total_pairs > 0:
                st.success(
                    f"**{total_pairs}** total pair{'s' if total_pairs != 1 else ''} to test "
                    f"({total_cross_pairs} cross-bucket, {total_intra_pairs} intra-bucket)."
                )
else:
    # Clear stale template state when file is removed
    for _k in ("_tpl_file_id", "_tpl_cross", "_tpl_intra", "_tpl_error"):
        st.session_state.pop(_k, None)

st.divider()


# ── Step 2: Run ───────────────────────────────────────────────────────────────
st.subheader("2  Run Analysis")

_total_pairs_preview = (
    sum(
        len(v) * (len(v) - 1) // 2
        for v in intra_var_selections.values()
        if len(v) >= 2
    ) + sum(
        len(var_meta[(var_meta["model"] == m) & (var_meta["bucket"] == b1)]) *
        len(var_meta[(var_meta["model"] == m) & (var_meta["bucket"] == b2)])
        for m, buckets in (
            {mdl: [bkt for _m, bkt in cross_bucket_selections if _m == mdl]
             for mdl in {_m for _m, _ in cross_bucket_selections}}.items()
        )
        for b1, b2 in combinations(buckets, 2)
    )
    if not var_meta.empty and cross_bucket_selections else 0
)
can_run = _total_pairs_preview > 0 or any(
    len(v) >= 2 for v in intra_var_selections.values()
)

if not can_run and uploaded_template is None:
    st.info("Upload a completed template above to enable the analysis.")
elif not can_run:
    st.warning("No valid pairs found in the uploaded template. Check your selections.")

if st.button("Run Synergy Analysis", type="primary", disabled=not can_run):
    all_results: list   = []
    prog                = st.progress(0, text="Starting…")
    total_y_cache: dict = {}
    _now_ts             = datetime.now().strftime("%Y%m%d_%H%M")

    # ── Output folder setup ───────────────────────────────────────────────────
    _save_folder = st.session_state.get("_out_folder", "").strip()
    _out_path    = None
    if _save_folder:
        try:
            _out_path = Path(_save_folder)
            _out_path.mkdir(parents=True, exist_ok=True)
        except Exception as _exc:
            st.warning(f"Cannot create output folder '{_save_folder}': {_exc}. Results will not be auto-saved.")

    # ── Build pairs grouped by model ──────────────────────────────────────────
    model_pairs: dict = defaultdict(list)  # model -> [(info1, info2, pair_type), ...]

    if cross_bucket_selections and not var_meta.empty:
        _m2b: dict = defaultdict(list)
        for model, bucket in cross_bucket_selections:
            _m2b[model].append(bucket)
        for model, buckets in _m2b.items():
            for b1, b2 in combinations(buckets, 2):
                vars1 = var_meta[(var_meta["model"] == model) & (var_meta["bucket"] == b1)]
                vars2 = var_meta[(var_meta["model"] == model) & (var_meta["bucket"] == b2)]
                for _, r1 in vars1.iterrows():
                    for _, r2 in vars2.iterrows():
                        model_pairs[model].append((
                            (model, r1["variable"],
                             str(r1.get("description", "") or "").strip(), b1),
                            (model, r2["variable"],
                             str(r2.get("description", "") or "").strip(), b2),
                            "cross_bucket",
                        ))

    if intra_var_selections:
        desc_lookup = {}
        bkt_lookup  = {}
        if not var_meta.empty:
            for _, row in var_meta.iterrows():
                key = (str(row["model"]).strip(), str(row["variable"]).strip())
                desc_lookup[key] = str(row.get("description", "") or "").strip()
                bkt_lookup[key]  = str(row.get("bucket",      "") or "").strip()
        for (model, bucket), vars_list in intra_var_selections.items():
            if len(vars_list) < 2:
                continue
            for v1, v2 in combinations(vars_list, 2):
                d1 = desc_lookup.get((model, v1), "")
                d2 = desc_lookup.get((model, v2), "")
                b1 = bkt_lookup.get((model, v1), bucket)
                b2 = bkt_lookup.get((model, v2), bucket)
                model_pairs[model].append((
                    (model, v1, d1, b1),
                    (model, v2, d2, b2),
                    "intra_bucket",
                ))

    total_pairs     = sum(len(p) for p in model_pairs.values())
    global_pair_idx = 0

    # ── Process model by model, saving after each ─────────────────────────────
    for model in sorted(model_pairs.keys()):
        model_results: list = []

        for (m1, id1, desc1, bkt1), (m2, id2, desc2, bkt2), pair_type in model_pairs[model]:
            global_pair_idx += 1
            prog.progress(
                global_pair_idx / max(total_pairs, 1),
                text=f"[{model}] {global_pair_idx}/{total_pairs}: {id1} × {id2}",
            )

            ts1     = _clip(get_series(wts,    m1, id1), period_start, period_end)
            ts2     = _clip(get_series(wts,    m2, id2), period_start, period_end)
            orig1_s = _clip(get_series(weekly, m1, id1), period_start, period_end)
            orig2_s = _clip(get_series(weekly, m2, id2), period_start, period_end)

            missing = []
            if ts1.empty: missing.append(f"support for '{id1}' in WeeklyTransformSupport")
            if ts2.empty: missing.append(f"support for '{id2}' in WeeklyTransformSupport")
            if missing:
                model_results.append({
                    "var1": id1, "var2": id2, "desc1": desc1, "desc2": desc2,
                    "bucket1": bkt1, "bucket2": bkt2,
                    "model1": m1, "model2": m2, "pair_type": pair_type,
                    "error": "Missing data: " + "; ".join(missing),
                    "is_significant": False,
                })
                continue

            if m1 not in total_y_cache:
                dep_var = model_dependents.get(m1)
                total_y_cache[m1] = _clip(
                    get_total_model_contributions(weekly, m1, dependent_var=dep_var),
                    period_start, period_end,
                )
            total_y = total_y_cache[m1]

            if total_y.empty or total_y.std() < 1e-6:
                model_results.append({
                    "var1": id1, "var2": id2, "desc1": desc1, "desc2": desc2,
                    "bucket1": bkt1, "bucket2": bkt2,
                    "model1": m1, "model2": m2, "pair_type": pair_type,
                    "error": f"No usable total contributions found for model '{m1}'.",
                    "is_significant": False,
                })
                continue

            res = compute_synergy_model(total_y, ts1, ts2, ci_level, n_bootstrap)
            res.update({
                "var1": id1, "var2": id2, "desc1": desc1, "desc2": desc2,
                "bucket1": bkt1, "bucket2": bkt2,
                "model1": m1, "model2": m2, "pair_type": pair_type,
            })

            if not res.get("error"):
                idx     = res["index"]
                orig_c1 = float(orig1_s.reindex(idx).fillna(0).sum())
                orig_c2 = float(orig2_s.reindex(idx).fillna(0).sum())
                res["orig_contrib1"] = orig_c1
                res["orig_contrib2"] = orig_c2

                c       = res["coefficients"]
                raw_A   = float(np.sum(res["support1"]        * c[0]))
                raw_B   = float(np.sum(res["support2"]        * c[1]))
                raw_syn = max(0.0, float(np.sum(res["synergy_support"] * c[2])))
                raw_tot = raw_A + raw_B + raw_syn
                combined = orig_c1 + orig_c2
                if raw_tot > 1e-12 and combined > 0:
                    res["adj_contrib1"]    = combined * raw_A   / raw_tot
                    res["adj_contrib2"]    = combined * raw_B   / raw_tot
                    res["synergy_contrib"] = combined * raw_syn / raw_tot
                else:
                    res["adj_contrib1"]    = orig_c1
                    res["adj_contrib2"]    = orig_c2
                    res["synergy_contrib"] = 0.0

            model_results.append(res)

        all_results.extend(model_results)

        # ── Save this model's results to folder ───────────────────────────────
        if _out_path is not None:
            _safe_m      = _safe_name(model)
            _m_cross_sig = [r for r in model_results if r.get("is_significant") and r.get("pair_type") == "cross_bucket"]
            _m_intra_sig = [r for r in model_results if r.get("is_significant") and r.get("pair_type") == "intra_bucket"]
            _n_sig       = len(_m_cross_sig) + len(_m_intra_sig)
            _save_errors = []

            for _res_list, _tag in [(_m_cross_sig, "bucket"), (_m_intra_sig, "variable")]:
                if not _res_list:
                    continue
                try:
                    (_out_path / f"synergy_{_tag}_{_safe_m}_{_now_ts}.xlsx").write_bytes(
                        export_to_excel(_res_list, selected_country).read()
                    )
                    (_out_path / f"synergy_{_tag}_{_safe_m}_{_now_ts}.pdf").write_bytes(
                        export_to_pdf(_res_list, selected_country).read()
                    )
                except Exception as _exc:
                    _save_errors.append(f"{_tag}: {_exc}")

            if _n_sig > 0 and not _save_errors:
                st.success(f"**{model}** — {_n_sig} synerg{'ies' if _n_sig != 1 else 'y'} saved to folder.")
            elif _n_sig == 0:
                st.info(f"**{model}** — no significant synergies found.")
            for _e in _save_errors:
                st.error(f"**{model}** save error — {_e}")

    prog.empty()
    st.session_state["all_results"]    = all_results
    st.session_state["result_country"] = selected_country
    st.session_state["result_period"]  = selected_period_name

st.divider()


# ── Step 3: Results ───────────────────────────────────────────────────────────
if (
    "all_results" in st.session_state
    and st.session_state.get("result_country") == selected_country
    and st.session_state.get("result_period")  == selected_period_name
):
    all_results  = st.session_state["all_results"]
    significant  = [r for r in all_results if r.get("is_significant")]
    tested_count = len(all_results)
    error_count  = sum(1 for r in all_results if r.get("error"))

    result_period = st.session_state.get("result_period", "Full range")
    st.subheader(f"3  Results  —  {result_period}")

    bcol1, bcol2, bcol3 = st.columns(3)
    bcol1.metric("Pairs Tested",     tested_count - error_count)
    bcol2.metric("Synergies Found",  len(significant))
    bcol3.metric("Errors / Skipped", error_count)

    if not significant:
        st.info(
            "No statistically significant synergies were found. "
            "Try selecting different buckets or lowering the confidence interval threshold."
        )
    else:
        significant     = sorted(significant, key=lambda r: r.get("delta_r2", 0), reverse=True)
        ci_pct          = int(ci_level * 100)
        cross_synergies = [r for r in significant if r.get("pair_type") == "cross_bucket"]
        intra_synergies = [r for r in significant if r.get("pair_type") == "intra_bucket"]

        # ── Label helper ──────────────────────────────────────────────────────
        def _var_label(var: str, desc: str, bucket: str) -> str:
            detail = var + (f" | {bucket}" if bucket else "")
            return f"{desc} ({detail})" if desc else var + (f" | {bucket}" if bucket else "")

        # ── Summary table ─────────────────────────────────────────────────────
        def _render_summary_table(res_list: list, anchor_prefix: str, hdr_color: str) -> None:
            rows_html = ""
            for idx, res in enumerate(res_list):
                anchor    = f"{anchor_prefix}-{idx}"
                lbl1      = _var_label(res["var1"], res.get("desc1", ""), res.get("bucket1", ""))
                lbl2      = _var_label(res["var2"], res.get("desc2", ""), res.get("bucket2", ""))
                pair_lbl  = f"{lbl1}  ×  {lbl2}"
                model_lbl = (
                    res["model1"] if res["model1"] == res["model2"]
                    else f"{res['model1']} / {res['model2']}"
                )
                rows_html += (
                    f"<tr>"
                    f"<td style='padding:6px 12px'>{idx+1}</td>"
                    f"<td style='padding:6px 12px'><a href='#{anchor}'>{pair_lbl}</a></td>"
                    f"<td style='padding:6px 12px'>{model_lbl}</td>"
                    f"<td style='padding:6px 12px'>{res.get('delta_r2', 0):.4f}</td>"
                    f"<td style='padding:6px 12px'>{res.get('r2_full',  0):.4f}</td>"
                    f"<td style='padding:6px 12px'>{res['coefficients'][2]:.4f}</td>"
                    f"<td style='padding:6px 12px'>{res.get('f_stat',   0):.2f}</td>"
                    f"<td style='padding:6px 12px'>{res.get('p_value',  1):.4f}</td>"
                    f"<td style='padding:6px 12px'>{res.get('synergy_formulation', '')}</td>"
                    f"</tr>"
                )
            st.markdown(
                f"""
                <table style='border-collapse:collapse; width:100%; font-size:0.88rem'>
                  <thead>
                    <tr style='background:{hdr_color}; color:white'>
                      <th style='padding:6px 12px'>#</th>
                      <th style='padding:6px 12px'>Pair (Variable | Bucket)</th>
                      <th style='padding:6px 12px'>Model</th>
                      <th style='padding:6px 12px'>Delta R²</th>
                      <th style='padding:6px 12px'>R² (full)</th>
                      <th style='padding:6px 12px'>Synergy Coeff</th>
                      <th style='padding:6px 12px'>F-stat</th>
                      <th style='padding:6px 12px'>p-value</th>
                      <th style='padding:6px 12px'>Formulation</th>
                    </tr>
                  </thead>
                  <tbody>{rows_html}</tbody>
                </table>
                """,
                unsafe_allow_html=True,
            )

        # ── Detail panels ─────────────────────────────────────────────────────
        def _render_detail_panels(res_list: list, anchor_prefix: str) -> None:
            for idx, res in enumerate(res_list):
                anchor     = f"{anchor_prefix}-{idx}"
                lbl1       = _var_label(res["var1"], res.get("desc1", ""), res.get("bucket1", ""))
                lbl2       = _var_label(res["var2"], res.get("desc2", ""), res.get("bucket2", ""))
                title      = f"{lbl1}  ×  {lbl2}"
                subtitle   = (
                    f"({res['model1']})" if res["model1"] == res["model2"]
                    else f"({res['model1']} / {res['model2']})"
                )
                result_key = f"{res['model1']}_{res['var1']}_{res['model2']}_{res['var2']}"

                st.markdown(f"<div id='{anchor}'></div>", unsafe_allow_html=True)
                with st.expander(f"{idx+1}.  {title}   {subtitle}", expanded=False):
                    b1 = res.get("bucket1", "")
                    b2 = res.get("bucket2", "")
                    st.caption(
                        f"**{res['var1']}** (Bucket: {b1})  ×  **{res['var2']}** (Bucket: {b2})"
                        f"  |  Formulation: {res['synergy_formulation']}"
                        f"  |  N = {res['n_obs']}  |  CI = {ci_pct}%"
                    )

                    mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
                    mc1.metric("R² Base",         f"{res['r2_base']:.4f}")
                    mc2.metric("R² with Synergy", f"{res['r2_full']:.4f}")
                    mc3.metric("Delta R²",         f"{res['delta_r2']:.4f}")
                    mc4.metric("Synergy Coeff",    f"{res['coefficients'][2]:.4f}")
                    mc5.metric("F-stat",           f"{res['f_stat']:.2f}")
                    mc6.metric("p-value",          f"{res['p_value']:.4f}")

                    # CI table
                    ci_df = pd.DataFrame({
                        "Variable":              [lbl1, lbl2, "Synergy"],
                        "Coefficient":           res["coefficients"],
                        f"CI Lower ({ci_pct}%)": res["ci_lower"],
                        f"CI Upper ({ci_pct}%)": res["ci_upper"],
                    }).set_index("Variable")
                    st.dataframe(ci_df.style.format("{:.6f}"), use_container_width=True)

                    # Contribution breakdown
                    st.markdown("**Contribution Breakdown** — sum over analysis period")
                    orig_c1 = res.get("orig_contrib1",  0.0)
                    orig_c2 = res.get("orig_contrib2",  0.0)
                    adj_c1  = res.get("adj_contrib1",   orig_c1)
                    adj_c2  = res.get("adj_contrib2",   orig_c2)
                    syn_cab = res.get("synergy_contrib", 0.0)

                    orig_total     = orig_c1 + orig_c2
                    adj_total      = adj_c1  + adj_c2
                    combined       = orig_c1 + orig_c2
                    adjustment_pct = (abs(adj_total - orig_total) / abs(orig_total) * 100) if orig_total != 0 else 0
                    c1_pct         = (adj_c1 / combined * 100) if combined != 0 else 0
                    c2_pct         = (adj_c2 / combined * 100) if combined != 0 else 0
                    small_contrib  = adj_c1 < 0.05 * combined or adj_c2 < 0.05 * combined

                    contrib_df = pd.DataFrame([
                        {"Description": f"Original contribution — {lbl1}",         "Value": orig_c1},
                        {"Description": f"Original contribution — {lbl2}",         "Value": orig_c2},
                        {"Description": f"Synergy-adjusted contribution — {lbl1}", "Value": adj_c1},
                        {"Description": f"Synergy-adjusted contribution — {lbl2}", "Value": adj_c2},
                        {"Description": "Synergy contribution",                     "Value": syn_cab},
                    ]).set_index("Description")
                    st.dataframe(contrib_df.style.format({"Value": "{:,.2f}"}), use_container_width=True)

                    # Raw model outputs
                    st.markdown("**Raw Model Outputs** (before contribution scaling)")
                    raw_c1  = res.get("raw_coeff1",  0.0)
                    raw_c2  = res.get("raw_coeff2",  0.0)
                    raw_syn = res.get("raw_synergy", 0.0)
                    raw_tot = res.get("raw_total",   1.0)
                    rp1     = (raw_c1  / raw_tot * 100) if raw_tot != 0 else 0
                    rp2     = (raw_c2  / raw_tot * 100) if raw_tot != 0 else 0
                    rp_syn  = (raw_syn / raw_tot * 100) if raw_tot != 0 else 0
                    raw_df  = pd.DataFrame([
                        {"Component": f"{lbl1} support × coefficient", "Sum": raw_c1,  "% of Total": rp1},
                        {"Component": f"{lbl2} support × coefficient", "Sum": raw_c2,  "% of Total": rp2},
                        {"Component": "Synergy support × coefficient",  "Sum": raw_syn, "% of Total": rp_syn},
                        {"Component": "TOTAL",                          "Sum": raw_tot, "% of Total": 100.0},
                    ]).set_index("Component")
                    st.dataframe(
                        raw_df.style.format({"Sum": "{:,.2f}", "% of Total": "{:.1f}%"}),
                        use_container_width=True,
                    )

                    if adjustment_pct > 20:
                        st.warning(
                            f"⚠️ **Large adjustment detected** ({adjustment_pct:.1f}% change from original). "
                            "This may indicate that one variable's support pattern doesn't align well "
                            "with the combined total."
                        )
                    if small_contrib:
                        lbl_small = lbl1 if adj_c1 < 0.05 * combined else lbl2
                        st.warning(
                            f"⚠️ **Small adjusted contribution** — {lbl_small} "
                            f"adjusted contribution is < 5% of combined total. "
                            f"({c1_pct:.1f}% and {c2_pct:.1f}%)"
                        )

                    # Chart
                    d1_disp = res.get("desc1") or res["var1"]
                    d2_disp = res.get("desc2") or res["var2"]
                    fig = create_synergy_chart(res, d1_disp, d2_disp)
                    st.plotly_chart(
                        fig, use_container_width=True,
                        key=f"chart_{anchor_prefix}_{result_key}",
                    )

                    # Weekly breakdown
                    st.markdown("**Weekly Breakdown**")
                    weekly_data = pd.DataFrame({
                        "Date":     res["index"],
                        lbl1:       res["support1"] * res["coefficients"][0],
                        lbl2:       res["support2"] * res["coefficients"][1],
                        "Synergy":  res["synergy_support"] * res["coefficients"][2],
                        "Combined": res["y_hat"],
                        "Actual":   res["y"],
                    })
                    st.dataframe(
                        weekly_data.style.format({
                            lbl1:       "{:,.2f}",
                            lbl2:       "{:,.2f}",
                            "Synergy":  "{:,.2f}",
                            "Combined": "{:,.2f}",
                            "Actual":   "{:,.2f}",
                        }),
                        use_container_width=True,
                        height=400,
                    )

        # ── Render cross-bucket results ───────────────────────────────────────
        if cross_synergies:
            st.markdown("#### Cross-Bucket Synergies")
            _render_summary_table(cross_synergies, "synergy-cross", "#2E4057")
            st.divider()
            st.markdown("### Cross-Bucket Details")
            _render_detail_panels(cross_synergies, "synergy-cross")

        # ── Render intra-bucket results ───────────────────────────────────────
        if intra_synergies:
            st.markdown("#### Intra-Bucket Synergies")
            _render_summary_table(intra_synergies, "synergy-intra", "#8B6F47")
            st.divider()
            st.markdown("### Intra-Bucket Details")
            _render_detail_panels(intra_synergies, "synergy-intra")


# ── Sidebar: Export (save to folder) ─────────────────────────────────────────
_cached_results = st.session_state.get("all_results", [])
_cross_sig      = [r for r in _cached_results if r.get("is_significant") and r.get("pair_type") == "cross_bucket"]
_intra_sig      = [r for r in _cached_results if r.get("is_significant") and r.get("pair_type") == "intra_bucket"]
_export_country = st.session_state.get("result_country", selected_country)


def _safe_name(s: str, max_len: int = 28) -> str:
    return re.sub(r"[^\w\-]", "_", s)[:max_len]


# Group significant results by model (needed both for summary display and on save)
_cross_by_model: dict = defaultdict(list)
for _r in _cross_sig:
    _cross_by_model[_r["model1"]].append(_r)

_intra_by_model: dict = defaultdict(list)
for _r in _intra_sig:
    _intra_by_model[_r["model1"]].append(_r)

_all_export_models = sorted(
    set(list(_cross_by_model.keys()) + list(_intra_by_model.keys()))
)

with st.sidebar:
    st.divider()
    st.subheader("Re-save Results")

    if not _cross_sig and not _intra_sig:
        st.caption("Run analysis to enable export.")
    else:
        for _model in _all_export_models:
            _m_cross = _cross_by_model.get(_model, [])
            _m_intra = _intra_by_model.get(_model, [])
            _m_total = len(_m_cross) + len(_m_intra)
            st.markdown(f"**{_model}**")
            st.caption(
                f"{_m_total} synerg{'ies' if _m_total != 1 else 'y'} "
                f"({len(_m_cross)} cross-bucket, {len(_m_intra)} intra-bucket)"
            )

        st.caption("Folder set above. Click to overwrite with latest results.")

        if st.button("Re-save to Folder", use_container_width=True):
            _resave_folder = st.session_state.get("_out_folder", "").strip()
            if not _resave_folder:
                st.error("Set an output folder above first.")
            else:
                _now_ts = datetime.now().strftime("%Y%m%d_%H%M")
                try:
                    _out_path = Path(_resave_folder)
                    _out_path.mkdir(parents=True, exist_ok=True)
                    _saved: list = []
                    _errs:  list = []
                    for _model in _all_export_models:
                        _safe_m  = _safe_name(_model)
                        _m_cross = _cross_by_model.get(_model, [])
                        _m_intra = _intra_by_model.get(_model, [])
                        for _res_list, _tag in [(_m_cross, "bucket"), (_m_intra, "variable")]:
                            if not _res_list:
                                continue
                            try:
                                (_out_path / f"synergy_{_tag}_{_safe_m}_{_now_ts}.xlsx").write_bytes(
                                    export_to_excel(_res_list, _export_country).read()
                                )
                                (_out_path / f"synergy_{_tag}_{_safe_m}_{_now_ts}.pdf").write_bytes(
                                    export_to_pdf(_res_list, _export_country).read()
                                )
                                _saved.append(f"{_model} ({_tag})")
                            except Exception as _exc:
                                _errs.append(f"{_model} ({_tag}): {_exc}")
                    if _saved:
                        st.success(f"Re-saved {len(_saved)} export(s).")
                    for _e in _errs:
                        st.error(f"Error — {_e}")
                except Exception as _exc:
                    st.error(f"Could not write to folder: {_exc}")
