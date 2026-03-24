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

    st.divider()
    st.caption("Estimation: NNLS — no intercept — no seasonality")
    st.caption("Only pairs with a statistically significant positive synergy coefficient are reported.")


# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading workbook…")
def _load(country: str):
    return load_country_data(country)


success, err_msg, data = _load(selected_country)

if not success:
    st.error(err_msg)
    st.stop()

weekly   = data.get("weekly", pd.DataFrame())
wts      = data.get("weekly_transform_support", pd.DataFrame())
var_meta = data.get("variable_meta", pd.DataFrame())

if weekly.empty:
    st.error("Sheet 'Weekly' not found or empty.")
    st.stop()
if wts.empty:
    st.error("Sheet 'WeeklyTransformSupport' not found or empty.")
    st.stop()

# ── Reporting period selector (added to sidebar after data is loaded) ─────────
reporting_periods = data.get("reporting_periods", [])
period_start: pd.Timestamp | None = None
period_end:   pd.Timestamp | None = None
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

fcol1, fcol2, fcol3 = st.columns([2, 2, 3])
with fcol1:
    all_models = ["All"] + sorted(catalogue["model"].unique().tolist())
    filter_model = st.selectbox("Filter by Model", all_models)
with fcol2:
    all_buckets = ["All"] + sorted(
        b for b in catalogue["bucket"].dropna().unique() if b
    )
    filter_bucket = st.selectbox("Filter by Bucket", all_buckets)
with fcol3:
    search_text = st.text_input("Search variable / description", placeholder="Type to filter…")

display_df = catalogue.copy()
if filter_model != "All":
    display_df = display_df[display_df["model"] == filter_model]
if filter_bucket != "All":
    display_df = display_df[display_df["bucket"] == filter_bucket]
if search_text:
    mask = (
        display_df["variable"].str.contains(search_text, case=False, na=False)
        | display_df.get("description", pd.Series("", index=display_df.index))
          .str.contains(search_text, case=False, na=False)
    )
    display_df = display_df[mask]

show_cols = ["Select", "model", "variable", "description", "bucket", "transform_summary"]
show_cols = [c for c in show_cols if c in display_df.columns]

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
        "transform_summary": st.column_config.TextColumn("Transformation",   width="large"),
    },
    disabled=["model", "variable", "description", "bucket", "transform_summary"],
    key=f"var_table_{selected_country}",
)

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
            sel_mv.append((r["model"], r["variable"], r.get("description", "") or ""))

    pairs = list(combinations(sel_mv, 2))
    all_results = []
    prog = st.progress(0, text="Starting…")

    # Cache total contributions per model to avoid recomputing
    total_y_cache: dict = {}

    for i, ((m1, v1, d1), (m2, v2, d2)) in enumerate(pairs):
        prog.progress(
            (i + 1) / len(pairs),
            text=f"Pair {i+1}/{len(pairs)}: {v1} x {v2}",
        )

        ts1 = _clip(get_series(wts, m1, v1), period_start, period_end)
        ts2 = _clip(get_series(wts, m2, v2), period_start, period_end)

        missing = []
        if ts1.empty: missing.append(f"support for '{v1}' in WeeklyTransformSupport")
        if ts2.empty: missing.append(f"support for '{v2}' in WeeklyTransformSupport")
        if missing:
            all_results.append({
                "var1": v1, "var2": v2, "desc1": d1, "desc2": d2,
                "model1": m1, "model2": m2,
                "error": "Missing data: " + "; ".join(missing),
                "is_significant": False,
            })
            continue

        # Use total model contributions of model1 as Y (if cross-model pair, use m1)
        if m1 not in total_y_cache:
            total_y_cache[m1] = _clip(
                get_total_model_contributions(weekly, m1), period_start, period_end
            )
        total_y = total_y_cache[m1]

        if total_y.empty or total_y.std() < 1e-6:
            all_results.append({
                "var1": v1, "var2": v2, "desc1": d1, "desc2": d2,
                "model1": m1, "model2": m2,
                "error": f"No usable total contributions found for model '{m1}'.",
                "is_significant": False,
            })
            continue

        res = compute_synergy_model(total_y, ts1, ts2, ci_level, n_bootstrap)
        res.update({"var1": v1, "var2": v2, "desc1": d1, "desc2": d2,
                    "model1": m1, "model2": m2})
        if not res.get("error"):
            # Original Weekly contributions aligned to the synergy analysis period
            idx = res["index"]
            orig1_s = _clip(get_series(weekly, m1, v1), period_start, period_end)
            orig2_s = _clip(get_series(weekly, m2, v2), period_start, period_end)
            res["orig_contrib1"] = float(orig1_s.reindex(idx).fillna(0).sum())
            res["orig_contrib2"] = float(orig2_s.reindex(idx).fillna(0).sum())
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

        # ── Summary table with anchor links ──────────────────────────────────
        st.markdown("#### Significant Synergy Pairs")

        rows_html = ""
        for idx, res in enumerate(significant):
            anchor   = f"synergy-pair-{idx}"
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

        # ── Detail panels ─────────────────────────────────────────────────────
        for idx, res in enumerate(significant):
            anchor = f"synergy-pair-{idx}"
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

            with st.expander(f"{idx+1}.  {title}   {subtitle}", expanded=False):
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
                c = res["coefficients"]
                syn_c1  = float(np.sum(res["support1"]       * c[0]))
                syn_c2  = float(np.sum(res["support2"]       * c[1]))
                syn_cab = float(np.sum(res["synergy_support"] * c[2]))
                orig_c1 = res.get("orig_contrib1", 0.0)
                orig_c2 = res.get("orig_contrib2", 0.0)
                contrib_df = pd.DataFrame([
                    {"Description": f"Original model contribution — {lbl1}",          "Value": orig_c1},
                    {"Description": f"Original model contribution — {lbl2}",          "Value": orig_c2},
                    {"Description": f"Synergy-adjusted contribution — {lbl1}",        "Value": syn_c1},
                    {"Description": f"Synergy-adjusted contribution — {lbl2}",        "Value": syn_c2},
                    {"Description": f"Synergy contribution — {d1} + {d2}",            "Value": syn_cab},
                ]).set_index("Description")
                st.dataframe(
                    contrib_df.style.format({"Value": "{:,.2f}"}),
                    use_container_width=True,
                )

                # ── Chart ─────────────────────────────────────────────────────
                fig = create_synergy_chart(res, d1, d2)
                st.plotly_chart(fig, use_container_width=True)

    # ── Step 4: Export (only if synergies found) ──────────────────────────────
    if significant:
        st.divider()
        st.subheader("4  Export")

        ecol1, ecol2 = st.columns(2)
        with ecol1:
            xlsx_data = export_to_excel(significant, selected_country)
            st.download_button(
                label="Download Excel",
                data=xlsx_data,
                file_name=f"synergy_{selected_country}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        with ecol2:
            pdf_data = export_to_pdf(significant, selected_country)
            st.download_button(
                label="Download PDF",
                data=pdf_data,
                file_name=f"synergy_{selected_country}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
