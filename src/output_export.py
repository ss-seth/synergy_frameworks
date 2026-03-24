"""
Export utilities: interactive Plotly charts, Excel workbook, PDF report.
"""

import io

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import xlsxwriter
from fpdf import FPDF

matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Interactive Plotly chart (Streamlit display)
# ---------------------------------------------------------------------------

def create_synergy_chart(result: dict, var1: str, var2: str) -> go.Figure:
    """
    Build an interactive Plotly figure showing:
      - Actual combined contributions (black)
      - Model fit (blue dashed)
      - Individual components toggled via legend
    """
    dates = result["index"]
    c = result["coefficients"]
    ci_pct = int(result["ci_level"] * 100)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dates, y=result["y"],
        name="Actual (Sum of Contributions)",
        line=dict(color="black", width=2.5),
        mode="lines",
    ))

    fig.add_trace(go.Scatter(
        x=dates, y=result["y_hat"],
        name=f"Model Fit  (R²={result.get('r2_full', result.get('r_squared', 0)):.4f})",
        line=dict(color="royalblue", width=2, dash="dash"),
        mode="lines",
    ))

    fig.add_trace(go.Scatter(
        x=dates, y=result["support1"] * c[0],
        name=f"{var1} Component",
        line=dict(color="seagreen", width=1.5),
        mode="lines",
        visible="legendonly",
    ))

    fig.add_trace(go.Scatter(
        x=dates, y=result["support2"] * c[1],
        name=f"{var2} Component",
        line=dict(color="darkorange", width=1.5),
        mode="lines",
        visible="legendonly",
    ))

    fig.add_trace(go.Scatter(
        x=dates, y=result["synergy_support"] * c[2],
        name="Synergy Component",
        line=dict(color="crimson", width=1.5),
        mode="lines",
        visible="legendonly",
    ))

    fig.update_layout(
        title=f"Synergy: {var1}  ×  {var2}",
        xaxis_title="Date",
        yaxis_title="Contribution",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        height=440,
    )

    return fig


# ---------------------------------------------------------------------------
# Matplotlib chart (for PDF embedding)
# ---------------------------------------------------------------------------

def _mpl_chart(result: dict, var1: str, var2: str) -> io.BytesIO:
    dates = result["index"]
    c = result["coefficients"]

    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(dates, result["y"], color="black", linewidth=2, label="Actual")
    ax.plot(dates, result["y_hat"], color="royalblue", linewidth=1.8,
            linestyle="--", label=f"Model Fit (R²={result.get('r2_full', result.get('r_squared', 0)):.4f})")
    ax.plot(dates, result["synergy_support"] * c[2], color="crimson",
            linewidth=1.4, linestyle=":", label="Synergy Component")

    ax.set_xlabel("Date", fontsize=9)
    ax.set_ylabel("Contribution", fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=8)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


# ---------------------------------------------------------------------------
# Excel export
# ---------------------------------------------------------------------------

def export_to_excel(results: list, country: str = "") -> io.BytesIO:
    buf = io.BytesIO()
    wb = xlsxwriter.Workbook(buf, {"in_memory": True})

    # --- formats ---
    hdr = wb.add_format({"bold": True, "bg_color": "#2E4057",
                          "font_color": "white", "border": 1, "align": "center"})
    num = wb.add_format({"border": 1, "num_format": "0.000000"})
    pct = wb.add_format({"border": 1, "num_format": "0.00%"})
    txt = wb.add_format({"border": 1})
    dt  = wb.add_format({"border": 1, "num_format": "yyyy-mm-dd"})
    ttl = wb.add_format({"bold": True, "font_size": 13})
    sub = wb.add_format({"bold": True, "font_size": 10})
    err = wb.add_format({"border": 1, "font_color": "red"})

    # ── Summary sheet ────────────────────────────────────────────────────────
    ws_sum = wb.add_worksheet("Summary")
    ws_sum.write(0, 0, f"Synergy Results — {country}", ttl)
    ws_sum.write(1, 0, f"Bootstrap CI applied to all coefficient estimates", sub)

    sum_headers = [
        "Pair", "Model 1", "Var 1", "Model 2", "Var 2",
        "Synergy Coeff", "Synergy CI Lower", "Synergy CI Upper",
        "R² Base", "R² with Synergy", "Delta R²",
        "F-stat", "p-value", "Formulation", "N Obs", "CI Level",
    ]
    for c_idx, h in enumerate(sum_headers):
        ws_sum.write(3, c_idx, h, hdr)

    for r_idx, res in enumerate(results):
        row = r_idx + 4
        if res.get("error"):
            ws_sum.write(row, 0, f"{res.get('var1','')} x {res.get('var2','')}", txt)
            ws_sum.write(row, 1, res["error"], err)
            continue
        ci_pct = res["ci_level"]
        ws_sum.write(row, 0,  f"{res['var1']} x {res['var2']}", txt)
        ws_sum.write(row, 1,  res["model1"],                    txt)
        ws_sum.write(row, 2,  res["var1"],                      txt)
        ws_sum.write(row, 3,  res["model2"],                    txt)
        ws_sum.write(row, 4,  res["var2"],                      txt)
        ws_sum.write(row, 5,  res["coefficients"][2],           num)
        ws_sum.write(row, 6,  res["ci_lower"][2],               num)
        ws_sum.write(row, 7,  res["ci_upper"][2],               num)
        ws_sum.write(row, 8,  res.get("r2_base",  0),           num)
        ws_sum.write(row, 9,  res.get("r2_full",  0),           num)
        ws_sum.write(row, 10, res.get("delta_r2", 0),           num)
        ws_sum.write(row, 11, res.get("f_stat",   0),           num)
        ws_sum.write(row, 12, res.get("p_value",  1),           num)
        ws_sum.write(row, 13, res.get("synergy_formulation",""),txt)
        ws_sum.write(row, 14, res["n_obs"],                     txt)
        ws_sum.write(row, 15, f"{int(ci_pct*100)}%",            txt)

    ws_sum.set_column(0, 0, 35)
    ws_sum.set_column(1, 4, 22)
    ws_sum.set_column(5, 12, 16)

    # ── Per-pair detail sheets ───────────────────────────────────────────────
    for res in results:
        if res.get("error"):
            continue

        # Sheet name: truncate to 31 chars (Excel limit)
        pair_name = f"{res['var1'][:13]}x{res['var2'][:13]}"[:31]
        # Ensure uniqueness by appending index if needed
        existing = [ws.get_name() for ws in wb.worksheets()]
        base = pair_name
        suffix = 1
        while pair_name in existing:
            pair_name = f"{base[:28]}_{suffix}"
            suffix += 1

        ws = wb.add_worksheet(pair_name)
        ci_pct = int(res["ci_level"] * 100)
        coeffs = res["coefficients"]

        # Header info
        ws.write(0, 0, f"{res['var1']} × {res['var2']}", ttl)
        ws.write(1, 0, f"Model 1: {res['model1']}  |  Model 2: {res['model2']}", sub)
        ws.write(2, 0,
                 f"R2_base={res.get('r2_base',0):.4f}  R2_full={res.get('r2_full',0):.4f}"
                 f"  dR2={res.get('delta_r2',0):.4f}  N={res['n_obs']}  CI={ci_pct}%"
                 f"  [{res.get('synergy_formulation','')}]", sub)

        # Coefficient table
        ws.write(4, 0, "Coefficient Summary", sub)
        ci_headers = ["Variable", "Coefficient", f"CI Lower ({ci_pct}%)", f"CI Upper ({ci_pct}%)"]
        for c_idx, h in enumerate(ci_headers):
            ws.write(5, c_idx, h, hdr)
        labels = [res["var1"], res["var2"], "Synergy"]
        for i, (lbl, coef, lo, hi) in enumerate(
            zip(labels, coeffs, res["ci_lower"], res["ci_upper"])
        ):
            ws.write(6 + i, 0, lbl, txt)
            ws.write(6 + i, 1, coef, num)
            ws.write(6 + i, 2, lo,   num)
            ws.write(6 + i, 3, hi,   num)

        # Contribution breakdown table
        cb_start = 11
        ws.write(cb_start, 0, "Contribution Breakdown (sum over analysis period)", sub)
        cb_headers = ["Description", "Value"]
        for c_idx, h in enumerate(cb_headers):
            ws.write(cb_start + 1, c_idx, h, hdr)

        orig_c1 = res.get("orig_contrib1",  0.0)
        orig_c2 = res.get("orig_contrib2",  0.0)
        adj_c1  = res.get("adj_contrib1",   orig_c1)
        adj_c2  = res.get("adj_contrib2",   orig_c2)
        syn_cab = res.get("synergy_contrib", 0.0)
        cb_rows = [
            (f"Original model contribution — {res['var1']}",         orig_c1),
            (f"Original model contribution — {res['var2']}",         orig_c2),
            (f"Synergy-adjusted contribution — {res['var1']}",       adj_c1),
            (f"Synergy-adjusted contribution — {res['var2']}",       adj_c2),
            (f"Synergy contribution — {res['var1']} + {res['var2']}", syn_cab),
        ]
        for i, (lbl, val) in enumerate(cb_rows):
            ws.write(cb_start + 2 + i, 0, lbl, txt)
            ws.write(cb_start + 2 + i, 1, val, num)

        # Time series table
        ts_start_row = cb_start + 2 + len(cb_rows) + 2
        ts_headers = [
            "Date", "Actual", "Fitted",
            f"{res['var1']} Component", f"{res['var2']} Component",
            "Synergy Component", "Residual",
        ]
        for c_idx, h in enumerate(ts_headers):
            ws.write(ts_start_row, c_idx, h, hdr)

        for r_idx, (d, act, fit, s1v, s2v, syn, resid) in enumerate(zip(
            res["index"],
            res["y"],
            res["y_hat"],
            res["support1"] * coeffs[0],
            res["support2"] * coeffs[1],
            res["synergy_support"] * coeffs[2],
            res["residuals"],
        )):
            row = ts_start_row + 1 + r_idx
            ws.write_datetime(row, 0, d.to_pydatetime().replace(tzinfo=None), dt)
            for c_idx, v in enumerate([act, fit, s1v, s2v, syn, resid], 1):
                ws.write(row, c_idx, float(v), num)

        ws.set_column(0, 0, 13)
        ws.set_column(1, 6, 18)

    wb.close()
    buf.seek(0)
    return buf


# ---------------------------------------------------------------------------
# PDF export
# ---------------------------------------------------------------------------

def _pdf_safe(text: str) -> str:
    """Replace characters unsupported by Helvetica with ASCII equivalents."""
    return (
        text.replace("\u2014", "-")   # em dash
            .replace("\u2013", "-")   # en dash
            .replace("\u00d7", "x")   # multiplication sign
            .replace("\u00b2", "2")   # superscript 2
    )


def export_to_pdf(results: list, country: str = "") -> io.BytesIO:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # ── Cover page ───────────────────────────────────────────────────────────
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 20)
    pdf.ln(20)
    pdf.cell(0, 12, "Synergy Calculation Results", ln=True, align="C")
    pdf.set_font("Helvetica", size=13)
    pdf.cell(0, 9, _pdf_safe(f"Country: {country}"), ln=True, align="C")
    pdf.ln(6)
    pdf.set_font("Helvetica", size=10)
    pdf.cell(0, 7,
             "Constrained OLS (NNLS) - no intercept - no seasonality - all coefficients >= 0",
             ln=True, align="C")
    pdf.cell(0, 7,
             "Confidence intervals via percentile bootstrap",
             ln=True, align="C")

    # ── Per-pair pages ───────────────────────────────────────────────────────
    for res in results:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 10, _pdf_safe(f"Synergy: {res.get('var1', '')}  x  {res.get('var2', '')}"), ln=True)

        if res.get("error"):
            pdf.set_font("Helvetica", size=10)
            pdf.set_text_color(200, 0, 0)
            pdf.cell(0, 8, _pdf_safe(f"Error: {res['error']}"), ln=True)
            pdf.set_text_color(0, 0, 0)
            continue

        ci_pct = int(res["ci_level"] * 100)
        pdf.set_font("Helvetica", size=10)
        pdf.cell(0, 7,
                 _pdf_safe(f"Model 1: {res['model1']}    Model 2: {res['model2']}"),
                 ln=True)
        pdf.cell(0, 7,
                 _pdf_safe(
                     f"R2_base={res.get('r2_base',0):.4f}  R2_full={res.get('r2_full',0):.4f}"
                     f"  dR2={res.get('delta_r2',0):.4f}  F={res.get('f_stat',0):.2f}"
                     f"  p={res.get('p_value',1):.4f}  N={res['n_obs']}  CI={ci_pct}%"
                 ),
                 ln=True)
        pdf.cell(0, 7, _pdf_safe(f"Formulation: {res.get('synergy_formulation','')}"), ln=True)
        pdf.ln(3)

        # Coefficient table
        pdf.set_font("Helvetica", "B", 10)
        col_w = [60, 38, 38, 38]
        for w, h in zip(col_w, ["Variable", "Coefficient",
                                  f"CI Lower ({ci_pct}%)", f"CI Upper ({ci_pct}%)"]):
            pdf.cell(w, 7, h, border=1)
        pdf.ln()

        pdf.set_font("Helvetica", size=10)
        labels = [res["var1"], res["var2"], "Synergy"]
        for lbl, coef, lo, hi in zip(
            labels, res["coefficients"], res["ci_lower"], res["ci_upper"]
        ):
            pdf.cell(col_w[0], 7, _pdf_safe(lbl[:35]), border=1)
            pdf.cell(col_w[1], 7, f"{coef:.6f}", border=1)
            pdf.cell(col_w[2], 7, f"{lo:.6f}",   border=1)
            pdf.cell(col_w[3], 7, f"{hi:.6f}",   border=1)
            pdf.ln()

        pdf.ln(4)

        # Contribution breakdown table
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 7, "Contribution Breakdown (sum over analysis period)", ln=True)
        cb_col_w = [130, 44]
        for w, h in zip(cb_col_w, ["Description", "Value"]):
            pdf.cell(w, 7, h, border=1)
        pdf.ln()

        pdf.set_font("Helvetica", size=9)
        orig_c1 = res.get("orig_contrib1",  0.0)
        orig_c2 = res.get("orig_contrib2",  0.0)
        adj_c1  = res.get("adj_contrib1",   orig_c1)
        adj_c2  = res.get("adj_contrib2",   orig_c2)
        syn_cab = res.get("synergy_contrib", 0.0)
        cb_rows = [
            (f"Original model contribution - {res['var1']}",          orig_c1),
            (f"Original model contribution - {res['var2']}",          orig_c2),
            (f"Synergy-adjusted contribution - {res['var1']}",        adj_c1),
            (f"Synergy-adjusted contribution - {res['var2']}",        adj_c2),
            (f"Synergy contribution - {res['var1']} + {res['var2']}", syn_cab),
        ]
        for lbl, val in cb_rows:
            pdf.cell(cb_col_w[0], 6, _pdf_safe(lbl[:65]), border=1)
            pdf.cell(cb_col_w[1], 6, f"{val:,.2f}", border=1)
            pdf.ln()

        pdf.ln(4)

        # Chart
        chart_buf = _mpl_chart(res, res["var1"], res["var2"])
        pdf.image(chart_buf, x=10, w=190)

    out = io.BytesIO()
    pdf.output(out)
    out.seek(0)
    return out
