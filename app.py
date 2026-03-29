"""
Synergy Calculation Tool — Streamlit app.

Workflow:
  1. Select country (dropdown)
  2. Browse and select variables (grouped table with model, variable, description, bucket)
  3. Configure CI level + bootstrap iterations
  4. Run analysis → one model per unique pair
  5. View significant synergies + download Excel / PDF
"""

from itertools import combinations

import numpy as np
import pandas as pd
import streamlit as st

from src.data_loader import (
    get_bucket_series,
    get_classification,
    get_countries,
    get_series,
    get_total_model_contributions,
    load_country_data,
    parse_transformation,
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

# Hide Streamlit's deploy button
st.markdown(
    "<style>[data-testid='stAppDeployButton'] {display: none;}</style>",
    unsafe_allow_html=True,
)

# ── Sidebar ──────────────────────────────────────────────────────────────────
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

    include_bucket_level = st.checkbox(
        "Include bucket-level synergies",
        value=False,
        help="When enabled, also computes synergies across buckets (summed contributions, avg/dominant transformation). Useful when variable-level data is sparse.",
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

weekly           = data.get("weekly", pd.DataFrame())
wts              = data.get("weekly_transform_support", pd.DataFrame())
var_meta         = data.get("variable_meta", pd.DataFrame())
model_dependents = data.get("model_dependents", {})

if weekly.empty:
    st.error("Sheet 'Weekly' not found or empty.")
    st.stop()
if wts.empty:
    st.error("Sheet 'WeeklyTransformSupport' not found or empty.")
    st.stop()

# ── Reporting period selector (added to sidebar after data is loaded) ─────────
reporting_periods = data.get("reporting_periods", [])
period_start = None
period_end   = None
selected_period_name = "Full range"

with st.sidebar:
    st.divider()
    st.subheader("Reporting Period")
    if reporting_periods:
        period_opts = ["Full range"] + [p["period"] for p in reporting_periods]
        selected_period_name = st.selectbox("Select period", period_opts, key="period_sel")
        if selected_period_name != "Full range":
            sel_p = next(p for p in reporting_periods if p["period"] == selected_period_name)
            period_start = sel_p["start"]
            period_end   = sel_p["end"]
            st.caption(
                f"{period_start.strftime('%d %b %Y')}  –  {period_end.strftime('%d %b %Y')}"
            )
    else:
        st.caption("No reporting periods found in Summary sheet — using full date range.")


def _clip(s: pd.Series, start, end) -> pd.Series:
    """Restrict a DatetimeIndex Series to [start, end] inclusive."""
    if start is not None:
        s = s[s.index >= start]
    if end is not None:
        s = s[s.index <= end]
    return s


# ── Build variable catalogue ──────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _build_catalogue(country: str) -> pd.DataFrame:
    mv = (
        weekly[["model", "variable"]]
        .drop_duplicates()
        .sort_values(["model", "variable"])
        .reset_index(drop=True)
    )

    # Add classification from weekly sheet if available
    if "classification" in weekly.columns:
        classifications = []
        for _, row in mv.iterrows():
            classification = get_classification(weekly, row["model"], row["variable"])
            classifications.append(classification or "—")
        mv["classification"] = classifications
    else:
        mv["classification"] = "—"

    if not var_meta.empty:
        meta_cols = ["model", "variable", "bucket", "description", "transformation"]
        available = [c for c in meta_cols if c in var_meta.columns]
        mv = mv.merge(
            var_meta[available].drop_duplicates(["model", "variable"]),
            on=["model", "variable"], how="left",
        )
    else:
        mv["bucket"] = ""
        mv["description"] = ""
        mv["transformation"] = ""

    def _fmt(t):
        if not t or pd.isna(t):
            return "-"
        info = parse_transformation(str(t))
        parts = []
        if info["adstock"] is not None:
            parts.append(f"Adstock={info['adstock']}, Power={info['power']}, Lag={info['lag']}")
        if info["rolling_avg"] is not None:
            parts.append(f"Rolling Avg {info['rolling_avg']}w")
        return ", ".join(parts) if parts else "-"

    mv["transform_summary"] = mv["transformation"].apply(_fmt)
    mv.insert(0, "Select", False)
    return mv


catalogue = _build_catalogue(selected_country)


# ── Main UI ───────────────────────────────────────────────────────────────────
st.header(f"Synergy Analysis — {selected_country}")
st.divider()

# ── Step 1: Variable selection ────────────────────────────────────────────────
st.subheader("1  Select Variables")
st.markdown(
    "Tick **Select** for each variable to include. "
    "All unique pairs will be tested — only those with a statistically significant "
    "positive synergy coefficient will be shown in the results."
)

fcol1, fcol2, fcol3, fcol4 = st.columns([2, 2, 2, 3])
with fcol1:
    all_models = ["All"] + sorted(catalogue["model"].unique().tolist())
    filter_model = st.selectbox("Filter by Model", all_models)
with fcol2:
    all_buckets = ["All"] + sorted(
        b for b in catalogue["bucket"].dropna().unique() if b
    )
    filter_bucket = st.selectbox("Filter by Bucket", all_buckets)
with fcol3:
    all_classifications = ["All"] + sorted(
        c for c in catalogue["classification"].dropna().unique() if c and c != "—"
    )
    filter_classification = st.selectbox("Filter by Classification", all_classifications)
with fcol4:
    search_text = st.text_input("Search variable / description", placeholder="Type to filter…")

display_df = catalogue.copy()
if filter_model != "All":
    display_df = display_df[display_df["model"] == filter_model]
if filter_bucket != "All":
    display_df = display_df[display_df["bucket"] == filter_bucket]
if filter_classification != "All":
    display_df = display_df[display_df["classification"] == filter_classification]
if search_text:
    mask = (
        display_df["variable"].str.contains(search_text, case=False, na=False)
        | display_df.get("description", pd.Series("", index=display_df.index))
          .str.contains(search_text, case=False, na=False)
    )
    display_df = display_df[mask]

# Initialize selection state in session
if "var_selection" not in st.session_state:
    st.session_state.var_selection = {}

show_cols = ["Select", "model", "variable", "description", "bucket", "classification", "transform_summary"]
show_cols = [c for c in show_cols if c in display_df.columns]

# Update display_df with current selections from session state
for idx, row in display_df.iterrows():
    var_key = (row["model"], row["variable"])
    display_df.loc[idx, "Select"] = st.session_state.var_selection.get(var_key, False)

# Quick action buttons
btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)
with btn_col1:
    if st.button("✓ Select All", use_container_width=True):
        for idx, row in catalogue.iterrows():
            var_key = (row["model"], row["variable"])
            st.session_state.var_selection[var_key] = True
        st.rerun()

with btn_col2:
    if st.button("✗ Deselect All", use_container_width=True):
        st.session_state.var_selection.clear()
        st.rerun()

if search_text or filter_model != "All" or filter_bucket != "All" or filter_classification != "All":
    filtered_count = len(display_df)
    with btn_col3:
        if st.button(f"✓ Select Filtered ({filtered_count})", use_container_width=True):
            for idx, row in display_df.iterrows():
                var_key = (row["model"], row["variable"])
                st.session_state.var_selection[var_key] = True
            st.rerun()

    with btn_col4:
        if st.button(f"✗ Deselect Filtered ({filtered_count})", use_container_width=True):
            for idx, row in display_df.iterrows():
                var_key = (row["model"], row["variable"])
                st.session_state.var_selection[var_key] = False
            st.rerun()

edited = st.data_editor(
    display_df[show_cols].reset_index(drop=True),
    use_container_width=True,
    hide_index=True,
    height=380,
    column_config={
        "Select":            st.column_config.CheckboxColumn("Select",       width="small"),
        "model":             st.column_config.TextColumn("Model",            width="medium"),
        "variable":          st.column_config.TextColumn("Variable",         width="medium"),
        "description":       st.column_config.TextColumn("Description",      width="large"),
        "bucket":            st.column_config.TextColumn("Bucket",           width="medium"),
        "classification":    st.column_config.TextColumn("Classification",   width="small"),
        "transform_summary": st.column_config.TextColumn("Transformation",   width="large"),
    },
    disabled=["model", "variable", "description", "bucket", "classification", "transform_summary"],
    key=f"var_table_{selected_country}",
)

# Sync selections from data_editor back to session state
for idx, row in display_df.iterrows():
    var_key = (row["model"], row["variable"])
    if idx < len(edited):
        st.session_state.var_selection[var_key] = bool(edited.iloc[idx]["Select"])

selected_rows = edited[edited["Select"] == True]
n_sel = len(selected_rows)

if n_sel >= 2:
    n_pairs = n_sel * (n_sel - 1) // 2
    st.success(
        f"**{n_sel}** variables selected → **{n_pairs}** pair{'s' if n_pairs != 1 else ''} "
        "will be tested. Only significant synergies will be shown."
    )
elif n_sel == 1:
    st.warning("Select at least **2** variables.")
else:
    st.info("Use the table above to select variables.")

st.divider()


# ── Step 2: Run ───────────────────────────────────────────────────────────────
st.subheader("2  Run Analysis")

if st.button("Run Synergy Analysis", type="primary", disabled=n_sel < 2):
    # Recover (model, variable, description) tuples
    sel_mv = []
    for _, row in selected_rows.iterrows():
        match = catalogue[
            (catalogue["variable"] == row["variable"]) &
            (catalogue["model"]    == row["model"])
        ]
        if not match.empty:
            r = match.iloc[0]
            sel_mv.append((r["model"], r["variable"], r.get("description", "") or "", r.get("bucket", "") or ""))

    # Build pairs: variable-level first, then bucket-level if enabled
    pairs = []
    pair_types = []  # track whether each pair is "variable" or "bucket"

    # Variable-level pairs
    variable_pairs = list(combinations(sel_mv, 2))
    for (m1, v1, d1, b1), (m2, v2, d2, b2) in variable_pairs:
        pairs.append(((m1, v1, d1), (m2, v2, d2)))
        pair_types.append("variable")

    # Bucket-level pairs (if enabled)
    if include_bucket_level:
        # Get unique (model, bucket) combinations from selected variables
        # Filter out variables without buckets (empty strings, None, or NaN)
        valid_buckets = []
        excluded_vars = []
        for m, v, d, b in sel_mv:
            # Check if bucket is valid (not empty, None, or NaN)
            if b and isinstance(b, str) and b.strip():
                valid_buckets.append((m, b))
            else:
                var_label = f"{v}" + (f" — {d}" if d else "")
                excluded_vars.append(var_label)

        if excluded_vars:
            var_list = "\n".join(f"• {var}" for var in sorted(set(excluded_vars)))
            st.info(
                f"⚠️ **Bucket-level analysis**: The following {len(excluded_vars)} variables don't have bucket assignments and will be excluded from bucket-level synergy calculations:\n\n{var_list}"
            )

        bucket_set = set(valid_buckets)
        bucket_list = list(bucket_set)
        bucket_pairs = list(combinations(bucket_list, 2))

        for (m1, b1), (m2, b2) in bucket_pairs:
            pairs.append(((m1, b1, b1), (m2, b2, b2)))  # Use bucket name as description
            pair_types.append("bucket")

    all_results = []
    prog = st.progress(0, text="Starting…")

    # Cache total contributions per model to avoid recomputing
    total_y_cache: dict = {}

    for pair_idx, (pair_data, pair_type) in enumerate(zip(pairs, pair_types)):
        (m1, id1, desc1), (m2, id2, desc2) = pair_data
        prog.progress(
            (pair_idx + 1) / len(pairs),
            text=f"Pair {pair_idx+1}/{len(pairs)}: {id1} x {id2} ({pair_type})",
        )

        # Get time series based on pair type
        if pair_type == "variable":
            v1, v2 = id1, id2
            ts1 = _clip(get_series(wts, m1, v1), period_start, period_end)
            ts2 = _clip(get_series(wts, m2, v2), period_start, period_end)
            orig1_s = _clip(get_series(weekly, m1, v1), period_start, period_end)
            orig2_s = _clip(get_series(weekly, m2, v2), period_start, period_end)
        else:  # bucket level
            b1, b2 = id1, id2
            ts1, _ = get_bucket_series(wts, m1, b1, var_meta)
            ts2, _ = get_bucket_series(wts, m2, b2, var_meta)
            orig1_s, _ = get_bucket_series(weekly, m1, b1, var_meta)
            orig2_s, _ = get_bucket_series(weekly, m2, b2, var_meta)
            ts1 = _clip(ts1, period_start, period_end)
            ts2 = _clip(ts2, period_start, period_end)
            orig1_s = _clip(orig1_s, period_start, period_end)
            orig2_s = _clip(orig2_s, period_start, period_end)

        # Check for missing data
        # Check for missing data
        missing = []
        if ts1.empty: missing.append(f"support for '{id1}' in WeeklyTransformSupport")
        if ts2.empty: missing.append(f"support for '{id2}' in WeeklyTransformSupport")
        if missing:
            all_results.append({
                "var1": id1, "var2": id2, "desc1": desc1, "desc2": desc2,
                "model1": m1, "model2": m2, "pair_type": pair_type,
                "error": "Missing data: " + "; ".join(missing),
                "is_significant": False,
            })
            continue

        # Use total model contributions of model1 as Y (if cross-model pair, use m1)
        if m1 not in total_y_cache:
            dep_var = model_dependents.get(m1)   # exclude dependent to avoid double-count
            total_y_cache[m1] = _clip(
                get_total_model_contributions(weekly, m1, dependent_var=dep_var),
                period_start, period_end,
            )
        total_y = total_y_cache[m1]

        if total_y.empty or total_y.std() < 1e-6:
            all_results.append({
                "var1": id1, "var2": id2, "desc1": desc1, "desc2": desc2,
                "model1": m1, "model2": m2, "pair_type": pair_type,
                "error": f"No usable total contributions found for model '{m1}'.",
                "is_significant": False,
            })
            continue

        res = compute_synergy_model(total_y, ts1, ts2, ci_level, n_bootstrap)
        res.update({"var1": id1, "var2": id2, "desc1": desc1, "desc2": desc2,
                    "model1": m1, "model2": m2, "pair_type": pair_type})
        if not res.get("error"):
            # Original contributions aligned to the synergy analysis period
            idx = res["index"]
            orig_c1 = float(orig1_s.reindex(idx).fillna(0).sum())
            orig_c2 = float(orig2_s.reindex(idx).fillna(0).sum())
            res["orig_contrib1"] = orig_c1
            res["orig_contrib2"] = orig_c2

            # Scale synergy model outputs to original contribution space.
            # The NNLS regresses *total* model Y on just 2 variables, so raw
            # coefficients × support are inflated (they try to explain the whole
            # total, not just A/B's slice).  Fix: use the relative proportions
            # that the model assigns to each component (A, B, synergy) and apply
            # those proportions to the original combined MMM contribution (C1+C2).
            # This guarantees adj_A + adj_B + synergy = C1 + C2 and keeps all
            # five rows on the same contribution scale.
            c = res["coefficients"]
            raw_A   = float(np.sum(res["support1"]        * c[0]))
            raw_B   = float(np.sum(res["support2"]        * c[1]))
            # Clip to zero: negative sum would mean variables are net out-of-phase
            # (already gated in is_significant, but clip here as a safety net)
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
        all_results.append(res)

    prog.empty()
    st.session_state["all_results"]      = all_results
    st.session_state["result_country"]   = selected_country
    st.session_state["result_period"]    = selected_period_name

st.divider()


# ── Step 3: Results ───────────────────────────────────────────────────────────
if (
    "all_results" in st.session_state
    and st.session_state.get("result_country") == selected_country
    and st.session_state.get("result_period") == selected_period_name
):
    all_results = st.session_state["all_results"]
    significant  = [r for r in all_results if r.get("is_significant")]
    tested_count = len(all_results)
    error_count  = sum(1 for r in all_results if r.get("error"))

    result_period = st.session_state.get("result_period", "Full range")
    st.subheader(f"3  Results  —  {result_period}")

    # Summary banner
    bcol1, bcol2, bcol3 = st.columns(3)
    bcol1.metric("Pairs Tested",        tested_count - error_count)
    bcol2.metric("Synergies Found",      len(significant))
    bcol3.metric("Errors / Skipped",     error_count)

    if not significant:
        st.info(
            "No statistically significant synergies were found in the selected pairs. "
            "Try selecting different variables or lowering the confidence interval threshold."
        )
    else:
        significant = sorted(significant, key=lambda r: r.get("delta_r2", 0), reverse=True)
        ci_pct = int(ci_level * 100)

        # Split results by pair type
        var_synergies = [r for r in significant if r.get("pair_type") != "bucket"]
        bucket_synergies = [r for r in significant if r.get("pair_type") == "bucket"]

        # ── VARIABLE-LEVEL SYNERGIES ──────────────────────────────────────────
        if var_synergies:
            st.markdown("#### Variable-Level Synergies")

            rows_html = ""
            for idx, res in enumerate(var_synergies):
                anchor   = f"synergy-pair-var-{idx}"
                d1 = res.get("desc1") or res["var1"]
                d2 = res.get("desc2") or res["var2"]
                pair_lbl = f"{d1} x {d2}"
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
                    f"<td style='padding:6px 12px'>{res.get('r2_full', 0):.4f}</td>"
                    f"<td style='padding:6px 12px'>{res['coefficients'][2]:.4f}</td>"
                    f"<td style='padding:6px 12px'>{res.get('f_stat', 0):.2f}</td>"
                    f"<td style='padding:6px 12px'>{res.get('p_value', 1):.4f}</td>"
                    f"<td style='padding:6px 12px'>{res.get('synergy_formulation','')}</td>"
                    f"</tr>"
                )

            st.markdown(
                f"""
                <table style='border-collapse:collapse; width:100%; font-size:0.88rem'>
                  <thead>
                    <tr style='background:#2E4057; color:white'>
                      <th style='padding:6px 12px'>#</th>
                      <th style='padding:6px 12px'>Pair</th>
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

            st.divider()

        # ── BUCKET-LEVEL SYNERGIES ────────────────────────────────────────────
        if bucket_synergies:
            st.markdown("#### Bucket-Level Synergies")

            rows_html = ""
            for idx, res in enumerate(bucket_synergies):
                anchor   = f"synergy-pair-bucket-{idx}"
                d1 = res.get("desc1") or res["var1"]
                d2 = res.get("desc2") or res["var2"]
                pair_lbl = f"{d1} x {d2}"
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
                    f"<td style='padding:6px 12px'>{res.get('r2_full', 0):.4f}</td>"
                    f"<td style='padding:6px 12px'>{res['coefficients'][2]:.4f}</td>"
                    f"<td style='padding:6px 12px'>{res.get('f_stat', 0):.2f}</td>"
                    f"<td style='padding:6px 12px'>{res.get('p_value', 1):.4f}</td>"
                    f"<td style='padding:6px 12px'>{res.get('synergy_formulation','')}</td>"
                    f"</tr>"
                )

            st.markdown(
                f"""
                <table style='border-collapse:collapse; width:100%; font-size:0.88rem'>
                  <thead>
                    <tr style='background:#8B6F47; color:white'>
                      <th style='padding:6px 12px'>#</th>
                      <th style='padding:6px 12px'>Pair</th>
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

            st.divider()

        # Initialize export selection state
        if "export_selection" not in st.session_state:
            st.session_state.export_selection = {}

        # ── Detail panels: Variable-Level ────────────────────────────────────────
        if var_synergies:
            st.markdown("### Variable-Level Details")
            for idx, res in enumerate(var_synergies):
                anchor = f"synergy-pair-var-{idx}"
                d1 = res.get("desc1") or res["var1"]
                d2 = res.get("desc2") or res["var2"]
                title  = f"{d1}  x  {d2}"
                subtitle = (
                    f"({res['model1']})"
                    if res["model1"] == res["model2"]
                    else f"({res['model1']} / {res['model2']})"
                )

                # Inject anchor so the summary table links land here
                st.markdown(f"<div id='{anchor}'></div>", unsafe_allow_html=True)

                # Create unique key for this result (stable, not dependent on sort order)
                result_key = f"{res['model1']}_{res['var1']}_{res['model2']}_{res['var2']}"

                # Create expander with export checkbox
                exp_col1, exp_col2 = st.columns([0.85, 0.15])
                with exp_col1:
                    expander = st.expander(f"{idx+1}.  {title}   {subtitle}", expanded=False)
                with exp_col2:
                    is_selected = st.checkbox(
                        "Export",
                        value=st.session_state.export_selection.get(result_key, False),
                        key=f"export_checkbox_{result_key}",
                        label_visibility="collapsed",
                    )
                    st.session_state.export_selection[result_key] = is_selected

                with expander:
                    # ── Key metrics ───────────────────────────────────────────────
                    mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
                    mc1.metric("R² Base",         f"{res['r2_base']:.4f}")
                    mc2.metric("R² with Synergy", f"{res['r2_full']:.4f}")
                    mc3.metric("Delta R²",         f"{res['delta_r2']:.4f}")
                    mc4.metric("Synergy Coeff",    f"{res['coefficients'][2]:.4f}")
                    mc5.metric("F-stat",           f"{res['f_stat']:.2f}")
                    mc6.metric("p-value",          f"{res['p_value']:.4f}")

                    st.caption(
                        f"**{res['var1']}** | **{res['var2']}**  "
                        f"|  Formulation: {res['synergy_formulation']}  "
                        f"|  N = {res['n_obs']}  |  CI = {ci_pct}%"
                    )

                    # ── CI table ──────────────────────────────────────────────────
                    lbl1 = f"{d1} ({res['var1']})"
                    lbl2 = f"{d2} ({res['var2']})"
                    ci_df = pd.DataFrame({
                        "Variable":              [lbl1, lbl2, "Synergy"],
                        "Coefficient":           res["coefficients"],
                        f"CI Lower ({ci_pct}%)": res["ci_lower"],
                        f"CI Upper ({ci_pct}%)": res["ci_upper"],
                    }).set_index("Variable")
                    st.dataframe(ci_df.style.format("{:.6f}"), use_container_width=True)

                    # ── Contribution breakdown ────────────────────────────────────
                    st.markdown("**Contribution Breakdown** — sum over analysis period")
                    orig_c1  = res.get("orig_contrib1",  0.0)
                    orig_c2  = res.get("orig_contrib2",  0.0)
                    adj_c1   = res.get("adj_contrib1",   orig_c1)
                    adj_c2   = res.get("adj_contrib2",   orig_c2)
                    syn_cab  = res.get("synergy_contrib", 0.0)

                    # Calculate adjustment magnitude
                    orig_total = orig_c1 + orig_c2
                    adj_total = adj_c1 + adj_c2
                    adjustment_pct = (abs(adj_total - orig_total) / abs(orig_total) * 100) if orig_total != 0 else 0

                    # Check if adjusted contributions became too small
                    combined = orig_c1 + orig_c2
                    c1_pct = (adj_c1 / combined * 100) if combined != 0 else 0
                    c2_pct = (adj_c2 / combined * 100) if combined != 0 else 0
                    small_contrib = adj_c1 < 0.05 * combined or adj_c2 < 0.05 * combined

                    contrib_df = pd.DataFrame([
                        {"Description": f"Original contribution — {lbl1}",                "Value": orig_c1},
                        {"Description": f"Original contribution — {lbl2}",                "Value": orig_c2},
                        {"Description": f"Synergy-adjusted contribution — {lbl1}",        "Value": adj_c1},
                        {"Description": f"Synergy-adjusted contribution — {lbl2}",        "Value": adj_c2},
                        {"Description": f"Synergy contribution",                           "Value": syn_cab},
                    ]).set_index("Description")
                    st.dataframe(
                        contrib_df.style.format({"Value": "{:,.2f}"}),
                        use_container_width=True,
                    )

                    # Show raw model outputs
                    st.markdown("**Raw Model Outputs** (before contribution scaling)")
                    raw_c1 = res.get("raw_coeff1", 0.0)
                    raw_c2 = res.get("raw_coeff2", 0.0)
                    raw_syn = res.get("raw_synergy", 0.0)
                    raw_tot = res.get("raw_total", 1.0)

                    raw_pct1 = (raw_c1 / raw_tot * 100) if raw_tot != 0 else 0
                    raw_pct2 = (raw_c2 / raw_tot * 100) if raw_tot != 0 else 0
                    raw_pct_syn = (raw_syn / raw_tot * 100) if raw_tot != 0 else 0

                    raw_df = pd.DataFrame([
                        {"Component": f"{d1} support × coefficient", "Sum": raw_c1, "% of Total": raw_pct1},
                        {"Component": f"{d2} support × coefficient", "Sum": raw_c2, "% of Total": raw_pct2},
                        {"Component": "Synergy support × coefficient", "Sum": raw_syn, "% of Total": raw_pct_syn},
                        {"Component": "TOTAL", "Sum": raw_tot, "% of Total": 100.0},
                    ]).set_index("Component")
                    st.dataframe(
                        raw_df.style.format({"Sum": "{:,.2f}", "% of Total": "{:.1f}%"}),
                        use_container_width=True,
                    )

                    # Warning if adjustment is significant
                    if adjustment_pct > 20:
                        st.warning(
                            f"⚠️ **Large adjustment detected** ({adjustment_pct:.1f}% change from original). "
                            f"This may indicate that one variable's support pattern doesn't align well with the combined total."
                        )

                    if small_contrib:
                        st.warning(
                            f"⚠️ **Small adjusted contribution** — {lbl1 if adj_c1 < 0.05 * combined else lbl2} "
                            f"adjusted contribution is < 5% of combined total. "
                            f"({c1_pct:.1f}% and {c2_pct:.1f}%)"
                        )

                    # ── Chart ─────────────────────────────────────────────────────
                    fig = create_synergy_chart(res, d1, d2)
                    st.plotly_chart(fig, use_container_width=True)

                    # ── Weekly Breakdown ──────────────────────────────────────────
                    st.markdown("**Weekly Breakdown**")
                    weekly_data = pd.DataFrame({
                        "Date": res["index"],
                        d1: res["support1"] * res["coefficients"][0],
                        d2: res["support2"] * res["coefficients"][1],
                        "Synergy": res["synergy_support"] * res["coefficients"][2],
                        "Combined": res["y_hat"],
                        "Actual": res["y"],
                    })
                    st.dataframe(
                        weekly_data.style.format({
                            d1: "{:,.2f}",
                            d2: "{:,.2f}",
                            "Synergy": "{:,.2f}",
                            "Combined": "{:,.2f}",
                            "Actual": "{:,.2f}",
                        }),
                        use_container_width=True,
                        height=400,
                    )

        # ── Detail panels: Bucket-Level ──────────────────────────────────────────
        if bucket_synergies:
            st.markdown("### Bucket-Level Details")
            for idx, res in enumerate(bucket_synergies):
                anchor = f"synergy-pair-bucket-{idx}"
                d1 = res.get("desc1") or res["var1"]
                d2 = res.get("desc2") or res["var2"]
                title  = f"{d1}  x  {d2}"
                subtitle = (
                    f"({res['model1']})"
                    if res["model1"] == res["model2"]
                    else f"({res['model1']} / {res['model2']})"
                )

                # Inject anchor so the summary table links land here
                st.markdown(f"<div id='{anchor}'></div>", unsafe_allow_html=True)

                # Create unique key for this result (stable, not dependent on sort order)
                result_key = f"{res['model1']}_{res['var1']}_{res['model2']}_{res['var2']}"

                # Create expander with export checkbox
                exp_col1, exp_col2 = st.columns([0.85, 0.15])
                with exp_col1:
                    expander = st.expander(f"{idx+1}.  {title}   {subtitle}", expanded=False)
                with exp_col2:
                    is_selected = st.checkbox(
                        "Export",
                        value=st.session_state.export_selection.get(result_key, False),
                        key=f"export_checkbox_{result_key}",
                        label_visibility="collapsed",
                    )
                    st.session_state.export_selection[result_key] = is_selected

                with expander:
                    # ── Key metrics ───────────────────────────────────────────────
                    mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
                    mc1.metric("R² Base",         f"{res['r2_base']:.4f}")
                    mc2.metric("R² with Synergy", f"{res['r2_full']:.4f}")
                    mc3.metric("Delta R²",         f"{res['delta_r2']:.4f}")
                    mc4.metric("Synergy Coeff",    f"{res['coefficients'][2]:.4f}")
                    mc5.metric("F-stat",           f"{res['f_stat']:.2f}")
                    mc6.metric("p-value",          f"{res['p_value']:.4f}")

                    st.caption(
                        f"**{res['var1']}** | **{res['var2']}**  "
                        f"|  Formulation: {res['synergy_formulation']}  "
                        f"|  N = {res['n_obs']}  |  CI = {ci_pct}%"
                    )

                    # ── CI table ──────────────────────────────────────────────────
                    lbl1 = f"{d1} ({res['var1']})"
                    lbl2 = f"{d2} ({res['var2']})"
                    ci_df = pd.DataFrame({
                        "Variable":              [lbl1, lbl2, "Synergy"],
                        "Coefficient":           res["coefficients"],
                        f"CI Lower ({ci_pct}%)": res["ci_lower"],
                        f"CI Upper ({ci_pct}%)": res["ci_upper"],
                    }).set_index("Variable")
                    st.dataframe(ci_df.style.format("{:.6f}"), use_container_width=True)

                    # ── Contribution breakdown ────────────────────────────────────
                    st.markdown("**Contribution Breakdown** — sum over analysis period")
                    orig_c1  = res.get("orig_contrib1",  0.0)
                    orig_c2  = res.get("orig_contrib2",  0.0)
                    adj_c1   = res.get("adj_contrib1",   orig_c1)
                    adj_c2   = res.get("adj_contrib2",   orig_c2)
                    syn_cab  = res.get("synergy_contrib", 0.0)

                    # Calculate adjustment magnitude
                    orig_total = orig_c1 + orig_c2
                    adj_total = adj_c1 + adj_c2
                    adjustment_pct = (abs(adj_total - orig_total) / abs(orig_total) * 100) if orig_total != 0 else 0

                    # Check if adjusted contributions became too small
                    combined = orig_c1 + orig_c2
                    c1_pct = (adj_c1 / combined * 100) if combined != 0 else 0
                    c2_pct = (adj_c2 / combined * 100) if combined != 0 else 0
                    small_contrib = adj_c1 < 0.05 * combined or adj_c2 < 0.05 * combined

                    contrib_df = pd.DataFrame([
                        {"Description": f"Original contribution — {lbl1}",                "Value": orig_c1},
                        {"Description": f"Original contribution — {lbl2}",                "Value": orig_c2},
                        {"Description": f"Synergy-adjusted contribution — {lbl1}",        "Value": adj_c1},
                        {"Description": f"Synergy-adjusted contribution — {lbl2}",        "Value": adj_c2},
                        {"Description": f"Synergy contribution",                           "Value": syn_cab},
                    ]).set_index("Description")
                    st.dataframe(
                        contrib_df.style.format({"Value": "{:,.2f}"}),
                        use_container_width=True,
                    )

                    # Show raw model outputs
                    st.markdown("**Raw Model Outputs** (before contribution scaling)")
                    raw_c1 = res.get("raw_coeff1", 0.0)
                    raw_c2 = res.get("raw_coeff2", 0.0)
                    raw_syn = res.get("raw_synergy", 0.0)
                    raw_tot = res.get("raw_total", 1.0)

                    raw_pct1 = (raw_c1 / raw_tot * 100) if raw_tot != 0 else 0
                    raw_pct2 = (raw_c2 / raw_tot * 100) if raw_tot != 0 else 0
                    raw_pct_syn = (raw_syn / raw_tot * 100) if raw_tot != 0 else 0

                    raw_df = pd.DataFrame([
                        {"Component": f"{d1} support × coefficient", "Sum": raw_c1, "% of Total": raw_pct1},
                        {"Component": f"{d2} support × coefficient", "Sum": raw_c2, "% of Total": raw_pct2},
                        {"Component": "Synergy support × coefficient", "Sum": raw_syn, "% of Total": raw_pct_syn},
                        {"Component": "TOTAL", "Sum": raw_tot, "% of Total": 100.0},
                    ]).set_index("Component")
                    st.dataframe(
                        raw_df.style.format({"Sum": "{:,.2f}", "% of Total": "{:.1f}%"}),
                        use_container_width=True,
                    )

                    # Warning if adjustment is significant
                    if adjustment_pct > 20:
                        st.warning(
                            f"⚠️ **Large adjustment detected** ({adjustment_pct:.1f}% change from original). "
                            f"This may indicate that one variable's support pattern doesn't align well with the combined total."
                        )

                    if small_contrib:
                        st.warning(
                            f"⚠️ **Small adjusted contribution** — {lbl1 if adj_c1 < 0.05 * combined else lbl2} "
                            f"adjusted contribution is < 5% of combined total. "
                            f"({c1_pct:.1f}% and {c2_pct:.1f}%)"
                        )

                    # ── Chart ─────────────────────────────────────────────────────
                    fig = create_synergy_chart(res, d1, d2)
                    st.plotly_chart(fig, use_container_width=True)

                    # ── Weekly Breakdown ──────────────────────────────────────────
                    st.markdown("**Weekly Breakdown**")
                    weekly_data = pd.DataFrame({
                        "Date": res["index"],
                        d1: res["support1"] * res["coefficients"][0],
                        d2: res["support2"] * res["coefficients"][1],
                        "Synergy": res["synergy_support"] * res["coefficients"][2],
                        "Combined": res["y_hat"],
                        "Actual": res["y"],
                    })
                    st.dataframe(
                        weekly_data.style.format({
                            d1: "{:,.2f}",
                            d2: "{:,.2f}",
                            "Synergy": "{:,.2f}",
                            "Combined": "{:,.2f}",
                            "Actual": "{:,.2f}",
                        }),
                        use_container_width=True,
                        height=400,
                    )


# ── Sidebar: Export (shown whenever significant results are available) ────────
_cached_results = st.session_state.get("all_results", [])
_sig_export     = [r for r in _cached_results if r.get("is_significant")]
_export_country = st.session_state.get("result_country", selected_country)

with st.sidebar:
    st.divider()
    st.subheader("Export Results")
    if _sig_export:
        # Get selected pairs for export
        _export_selection = st.session_state.get("export_selection", {})
        _selected_pairs = []
        for res in _sig_export:
            result_key = f"{res['model1']}_{res['var1']}_{res['model2']}_{res['var2']}"
            if _export_selection.get(result_key, False):
                _selected_pairs.append(res)

        if _selected_pairs:
            st.caption(f"**{len(_selected_pairs)}** of **{len(_sig_export)}** pairs selected for export")

            xlsx_data = export_to_excel(_selected_pairs, _export_country)
            st.download_button(
                label="📊 Download Excel",
                data=xlsx_data,
                file_name=f"synergy_{_export_country}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
            pdf_data = export_to_pdf(_selected_pairs, _export_country)
            st.download_button(
                label="📄 Download PDF",
                data=pdf_data,
                file_name=f"synergy_{_export_country}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        else:
            st.caption(f"Select synergy pairs above to enable export ({len(_sig_export)} available)")
    else:
        st.caption("Run analysis to enable export.")
