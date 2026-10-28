from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
import math
import re
import requests
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, r2_score
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
)

APP_TITLE = "Hispanic Heritage Historical Data Explorer"
WB_BASE = "https://api.worldbank.org/v2"
WB_INDICATOR = "SP.DYN.LE00.IN"
INDICATOR_LABEL = "Life expectancy at birth"
UNITS = "years"
INDICATOR_DEFINITION = (
    "The average number of years a newborn would live if the mortality patterns "
    "observed at birth remained constant throughout the person's life."
)
COUNTRIES = {
    "Mexico": "MEX",
    "Brazil": "BRA",
    "Argentina": "ARG",
    "Colombia": "COL",
    "Chile": "CHL",
    "Peru": "PER",
    "Venezuela": "VEN",
    "Ecuador": "ECU",
    "Uruguay": "URY",
}
DATA_URL = "https://data.worldbank.org/indicator/SP.DYN.LE00.IN"
API_URL_TEMPLATE = (
    "https://api.worldbank.org/v2/country/{code}/indicator/"
    "SP.DYN.LE00.IN?format=json&date={start}:{end}&per_page=1000"
)

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption("A student research tool for exploring historical life expectancy data and polynomial models.")

st.info(
    "Heritage context: Mexico and Argentina are Spanish-speaking countries. Brazil is included "
    "in the country menu only as a Latin American comparison; Portuguese is its official language, "
    "so Brazil is not a Spanish-speaking Hispanic country."
)

@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def fetch_world_bank(country_name: str, start_year: int, end_year: int) -> tuple[pd.DataFrame, str]:
    """Fetch genuine World Bank indicator values. Returns data and a user-readable status."""
    code = COUNTRIES[country_name]
    url = API_URL_TEMPLATE.format(code=code, start=start_year, end=end_year)
    try:
        response = requests.get(url, timeout=20, headers={"User-Agent": "Hispanic-Heritage-Student-Project/1.0"})
        response.raise_for_status()
        payload = response.json()
    except requests.Timeout:
        return pd.DataFrame(columns=["year", "value"]), "Request timed out. Check your connection and retry."
    except requests.RequestException as exc:
        return pd.DataFrame(columns=["year", "value"]), f"World Bank request failed: {exc}"
    except ValueError:
        return pd.DataFrame(columns=["year", "value"]), "The API response was not valid JSON."

    if not isinstance(payload, list) or len(payload) < 2 or not isinstance(payload[1], list):
        return pd.DataFrame(columns=["year", "value"]), "The API returned no usable observation list."
    rows = []
    for item in payload[1]:
        if not isinstance(item, dict):
            continue
        try:
            year = int(item["date"])
        except (KeyError, TypeError, ValueError):
            continue
        raw_value = item.get("value")
        if raw_value is None:
            continue
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            rows.append({"year": year, "value": value})
    frame = pd.DataFrame(rows, columns=["year", "value"])
    if frame.empty:
        return frame, "The World Bank API returned no non-missing values for this period."
    frame = frame.drop_duplicates(subset=["year"]).sort_values("year").reset_index(drop=True)
    return frame, "Live data retrieved from the World Bank API."

def clean_working_data(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Validate editable year/value data; reject invalid rows with clear reasons."""
    accepted, rejected = [], []
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["year", "value"]), ["No rows to validate."]
    seen = set()
    for idx, row in frame.iterrows():
        raw_year, raw_value = row.get("year"), row.get("value")
        try:
            if pd.isna(raw_year) or pd.isna(raw_value):
                rejected.append(f"Row {idx + 1}: missing year or value.")
                continue
            year_float = float(raw_year)
            if not year_float.is_integer():
                rejected.append(f"Row {idx + 1}: year must be an integer.")
                continue
            year = int(year_float)
            value = float(raw_value)
            if not math.isfinite(value):
                rejected.append(f"Row {idx + 1}: value must be finite.")
                continue
            if year < 1900 or year > datetime.now().year:
                rejected.append(f"Row {idx + 1}: year must be from 1900 through the current year.")
                continue
            if year in seen:
                rejected.append(f"Row {idx + 1}: duplicate year {year}.")
                continue
            if not (0 < value <= 120):
                rejected.append(f"Row {idx + 1}: life expectancy must be greater than 0 and no more than 120 years.")
                continue
            seen.add(year)
            accepted.append({"year": year, "value": value})
        except (TypeError, ValueError):
            rejected.append(f"Row {idx + 1}: year/value must be numeric.")
    cleaned = pd.DataFrame(accepted, columns=["year", "value"])
    if not cleaned.empty:
        cleaned = cleaned.sort_values("year").reset_index(drop=True)
    return cleaned, rejected

def fit_polynomial(data: pd.DataFrame, degree: int):
    """Fit y = polynomial(t), with t = year - base_year."""
    if data.empty:
        raise ValueError("There are no valid observations to fit.")
    if data["year"].nunique() < degree + 1:
        raise ValueError(f"Degree {degree} requires at least {degree + 1} distinct years; "
                         f"only {data['year'].nunique()} are available.")
    base_year = int(data["year"].min())
    t = data["year"].to_numpy(dtype=float) - base_year
    y = data["value"].to_numpy(dtype=float)
    coeff = np.polyfit(t, y, degree)
    pred = np.polyval(coeff, t)
    mae = float(mean_absolute_error(y, pred))
    r2 = float(r2_score(y, pred)) if len(y) > 1 else float("nan")
    residuals = y - pred
    return {"base_year": base_year, "coeff": coeff, "pred": pred, "mae": mae,
            "r2": r2, "residuals": residuals, "n": len(data),
            "min_year": int(data["year"].min()), "max_year": int(data["year"].max()),
            "mean": float(np.mean(y))}

def predict(model, years):
    arr = np.asarray(years, dtype=float)
    t = arr - model["base_year"]
    return np.polyval(model["coeff"], t)

def equation_text(model, precision=7):
    terms = []
    degree = len(model["coeff"]) - 1
    for i, c in enumerate(model["coeff"]):
        power = degree - i
        if abs(c) < 1e-14:
            continue
        c_text = f"{c:.{precision}g}"
        if power == 0:
            term = c_text
        elif power == 1:
            term = f"{c_text}t"
        else:
            term = f"{c_text}t^{power}"
        terms.append(term)
    return "y = " + (" + ".join(terms).replace("+ -", "- ") if terms else "0")

def classify_year(year, model, observed_years):
    if year in observed_years:
        return "Fitted value at an observed year"
    if model["min_year"] < year < model["max_year"]:
        return "Within historical range (interpolation)"
    if year < model["min_year"] or year > model["max_year"]:
        return "Outside fitted range (extrapolation)"
    return "Within fitted range"

def make_chart(data, model, country, comparisons, sample_increment, horizon, estimates=None, print_mode=False):
    fig, ax = plt.subplots(figsize=(10, 5.2) if not print_mode else (10, 5.8))
    ax.scatter(data["year"], data["value"], label=f"{country}: observed", s=24, zorder=3)
    smooth_years = np.linspace(model["min_year"], model["max_year"] + horizon, 500)
    smooth_vals = predict(model, smooth_years)
    historical = smooth_years <= model["max_year"]
    if historical.any():
        ax.plot(smooth_years[historical], smooth_vals[historical], label="Polynomial fit (historical range)", linewidth=2)
    if (~historical).any():
        ax.plot(smooth_years[~historical], smooth_vals[~historical], linestyle="--", label="Extrapolated model segment", linewidth=2)
    sampled_years = np.arange(model["min_year"], model["max_year"] + 1, sample_increment)
    ax.scatter(sampled_years, predict(model, sampled_years), marker="x", s=22, label=f"Sampled model markers (every {sample_increment} year(s))")
    ax.axvline(model["max_year"], linestyle=":", linewidth=1.4, label="End of observed fitted range")
    for comp_name, comp_df in comparisons.items():
        if not comp_df.empty:
            ax.plot(comp_df["year"], comp_df["value"], marker="o", markersize=3,
                    linewidth=1.2, label=f"{comp_name}: observed")
    if estimates:
        for year, value, label in estimates:
            marker = "D" if "Outside" in label else "s"
            ax.scatter([year], [value], marker=marker, s=65, zorder=5, label=f"Estimate {year}: {label}")
    ax.set_title(f"{country} — {INDICATOR_LABEL}")
    ax.set_xlabel("Year")
    ax.set_ylabel(f"Life expectancy ({UNITS})")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    return fig

def build_csv(data, country, edited):
    out = data.copy()
    out["country"] = country
    out["indicator"] = INDICATOR_LABEL
    out["indicator_code"] = WB_INDICATOR
    out["units"] = UNITS
    out["data_status"] = "user-modified scenario" if edited else "source data (not edited)"
    out["source"] = DATA_URL
    out["retrieved_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return out.to_csv(index=False).encode("utf-8")

def build_xlsx(data, summary, results, residuals, comparisons, edited):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        pd.DataFrame([summary]).to_excel(writer, sheet_name="Summary", index=False)
        data.to_excel(writer, sheet_name="Working Data", index=False)
        if results is not None and not results.empty:
            results.to_excel(writer, sheet_name="Calculated Years", index=False)
        residuals.to_excel(writer, sheet_name="Residuals", index=False)
        if comparisons:
            comp_rows = []
            for name, frame in comparisons.items():
                for _, row in frame.iterrows():
                    comp_rows.append({"country": name, "year": int(row["year"]), "value": float(row["value"])})
            if comp_rows:
                pd.DataFrame(comp_rows).to_excel(writer, sheet_name="Comparison Data", index=False)
        workbook = writer.book
        for sheet in writer.sheets.values():
            sheet.freeze_panes(1, 0)
            sheet.set_column(0, 12, 18)
    return output.getvalue()

def build_pdf(data, summary, results, chart_fig, findings, edited):
    output = BytesIO()
    doc = SimpleDocTemplate(output, pagesize=landscape(letter), rightMargin=0.45*inch,
                            leftMargin=0.45*inch, topMargin=0.4*inch, bottomMargin=0.4*inch)
    styles = getSampleStyleSheet()
    story = [Paragraph(APP_TITLE, styles["Title"]),
             Paragraph("Life expectancy at birth: Mexico and Argentina research", styles["Heading2"]),
             Spacer(1, 8)]
    meta = [
        ["Country", summary["Country"], "Indicator", summary["Indicator"]],
        ["Indicator code", summary["Indicator code"], "Units", summary["Units"]],
        ["Source", summary["Source"], "Retrieved (UTC)", summary["Retrieved (UTC)"]],
        ["Coverage", summary["Coverage"], "Observations", str(summary["Observations"])],
        ["Data status", summary["Data status"], "Polynomial degree", str(summary["Degree"])],
        ["Base year", str(summary["Base year"]), "Equation", summary["Equation"]],
        ["In-sample R²", summary["R-squared (in-sample)"], "In-sample MAE", summary["MAE (years)"]],
    ]
    table = Table(meta, colWidths=[1.0*inch, 3.15*inch, 1.15*inch, 4.1*inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#e8eef5")),
        ("GRID", (0,0), (-1,-1), 0.35, colors.grey),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("FONTSIZE", (0,0), (-1,-1), 7),
        ("LEFTPADDING", (0,0), (-1,-1), 4),
        ("RIGHTPADDING", (0,0), (-1,-1), 4),
    ]))
    story.extend([table, Spacer(1, 8)])
    chart_bytes = BytesIO()
    chart_fig.savefig(chart_bytes, format="png", dpi=150, bbox_inches="tight")
    chart_bytes.seek(0)
    story.append(Image(chart_bytes, width=9.4*inch, height=4.6*inch))
    story.append(Paragraph("Findings and limitations", styles["Heading2"]))
    for item in findings:
        story.append(Paragraph("• " + item, styles["BodyText"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "Life expectancy is a population-level period indicator, not a prediction of how long any particular person will live. "
        "The polynomial is descriptive; extrapolations may be implausible. MAE and in-sample R² do not prove forecasting accuracy. "
        "No statistically validated prediction intervals are provided.", styles["BodyText"]))
    story.append(PageBreak())
    story.append(Paragraph("Data preview", styles["Heading2"]))
    preview = [["Year", "Life expectancy (years)"]]
    for _, row in data.head(35).iterrows():
        preview.append([str(int(row["year"])), f'{float(row["value"]):.4f}'])
    dt = Table(preview, colWidths=[1.4*inch, 2.5*inch], repeatRows=1)
    dt.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#e8eef5")),
        ("GRID", (0,0), (-1,-1), 0.3, colors.grey), ("FONTSIZE", (0,0), (-1,-1), 8)
    ]))
    story.append(dt)
    if results is not None and not results.empty:
        story.append(Spacer(1, 12))
        story.append(Paragraph("Calculated-year results", styles["Heading2"]))
        rdata = [list(results.columns)] + results.astype(str).values.tolist()
        rt = Table(rdata, repeatRows=1)
        rt.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#e8eef5")),
            ("GRID", (0,0), (-1,-1), 0.3, colors.grey), ("FONTSIZE", (0,0), (-1,-1), 7)
        ]))
        story.append(rt)
    doc.build(story)
    return output.getvalue()

now_year = datetime.now().year
start_year, end_year = now_year - 70, now_year - 1
st.sidebar.header("Explorer controls")
country = st.sidebar.selectbox("Main country", list(COUNTRIES.keys()), index=list(COUNTRIES).index("Mexico"))
degree = st.sidebar.slider("Polynomial degree", 3, 8, 3)
sample_increment = st.sidebar.slider("Graph sample increment (years)", 1, 10, 1)
horizon = st.sidebar.slider("Forward display horizon (years)", 0, 50, 5)
comparison_choices = [name for name in COUNTRIES if name != country]
default_comparisons = [x for x in ["Argentina", "Mexico"] if x != country]
comparison_countries = st.sidebar.multiselect("Compare same indicator with", comparison_choices, default=default_comparisons)
print_mode = st.sidebar.toggle("Printer-friendly view", value=False)

st.write(f"**Research question:** How has life expectancy at birth changed over time in Mexico and Argentina, and what can polynomial models reveal—and fail to reveal—about these trends?")
st.caption(f"Requested search window: {start_year}–{end_year} (70 calendar years). The actual coverage depends on available official data.")

with st.spinner(f"Retrieving World Bank data for {country}..."):
    source_data, fetch_status = fetch_world_bank(country, start_year, end_year)
retrieved_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
st.caption(f"Source status: {fetch_status} | Retrieval timestamp (UTC): {retrieved_at}")
st.markdown(f"**Indicator:** {INDICATOR_LABEL} (`{WB_INDICATOR}`) — {UNITS}. {INDICATOR_DEFINITION}")
st.markdown(f"**Source:** [World Bank indicator page]({DATA_URL}) · [World Bank API documentation](https://datahelpdesk.worldbank.org/knowledgebase/articles/898581-api-basic-call-structures)")

if source_data.empty:
    st.error("No usable source observations are available right now. The app will not invent replacement data. Retry when the World Bank API is reachable.")
    st.stop()

if "active_country" not in st.session_state or st.session_state.active_country != country:
    st.session_state.active_country = country
    st.session_state.source_snapshot = source_data.copy()
    st.session_state.working_data = source_data.copy()
    st.session_state.calc_results = pd.DataFrame()
    st.session_state.last_model_key = None
elif "source_snapshot" not in st.session_state:
    st.session_state.source_snapshot = source_data.copy()
    st.session_state.working_data = source_data.copy()

left, right = st.columns([1.15, 1])
with left:
    st.subheader("Working data table")
    st.caption("Edit the year/value cells, then click Apply edits. Original source data remain available through Reset to source.")
    edited_table = st.data_editor(
        st.session_state.working_data,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "year": st.column_config.NumberColumn("Year", format="%d", step=1),
            "value": st.column_config.NumberColumn("Life expectancy (years)", format="%.4f", min_value=0.0001, max_value=120.0)
        },
        key=f"editor_{country}"
    )
    b1, b2 = st.columns(2)
    with b1:
        if st.button("Apply edits", use_container_width=True):
            cleaned, rejected = clean_working_data(edited_table)
            if cleaned.empty:
                st.error("No valid rows remain. Existing working data were not replaced.")
            else:
                st.session_state.working_data = cleaned
                st.session_state.calc_results = pd.DataFrame()
                st.session_state.last_model_key = None
                st.success(f"Applied {len(cleaned)} valid observations.")
                if rejected:
                    st.warning("Rejected rows: " + " ".join(rejected[:12]))
                    if len(rejected) > 12:
                        st.warning(f"{len(rejected)-12} additional rejection(s) not shown.")
                st.rerun()
    with b2:
        if st.button("Reset to source", use_container_width=True):
            st.session_state.working_data = st.session_state.source_snapshot.copy()
            st.session_state.calc_results = pd.DataFrame()
            st.session_state.last_model_key = None
            st.rerun()

data, rejected_now = clean_working_data(st.session_state.working_data)
edited = not data.equals(st.session_state.source_snapshot.sort_values("year").reset_index(drop=True))
data_status = "user-modified scenario" if edited else "source data (not edited)"
if data.empty:
    st.error("The working data contain no valid observations. Apply valid data or reset to source.")
    st.stop()
if rejected_now:
    st.warning("Validation note: " + " ".join(rejected_now[:8]))

with right:
    st.subheader("Coverage and summary")
    st.metric("Valid observations", len(data))
    st.metric("Available year range", f'{int(data["year"].min())}–{int(data["year"].max())}')
    st.metric("Mean life expectancy", f'{data["value"].mean():.3f} years')
    st.caption(f"Data status: {data_status}. API null values are omitted, not replaced.")

try:
    model = fit_polynomial(data, degree)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

model_key = (country, degree, tuple(data["year"].tolist()), tuple(np.round(data["value"].to_numpy(), 8)))
if st.session_state.get("last_model_key") != model_key:
    st.session_state.calc_results = pd.DataFrame()
    st.session_state.last_model_key = model_key

st.subheader("Polynomial model")
metrics = st.columns(5)
metrics[0].metric("Degree", degree)
metrics[1].metric("Base year", model["base_year"])
metrics[2].metric("In-sample R²", f'{model["r2"]:.5f}')
metrics[3].metric("In-sample MAE", f'{model["mae"]:.4f} years')
metrics[4].metric("Mean", f'{model["mean"]:.3f} years')
st.code(f"t = year - {model['base_year']}\n{equation_text(model)}")
st.caption(f"Fitted domain: {model['min_year']}–{model['max_year']}. The equation uses t = year − base year. Fit metrics are in-sample descriptive metrics, not proof of forecast accuracy.")

st.subheader("Graph and country comparison")
comparisons = {}
for comp in comparison_countries:
    with st.spinner(f"Retrieving comparison data for {comp}..."):
        comp_df, comp_status = fetch_world_bank(comp, start_year, end_year)
    comparisons[comp] = comp_df
    if comp_df.empty:
        st.warning(f"{comp}: {comp_status}")
    else:
        st.caption(f"{comp}: coverage {int(comp_df['year'].min())}–{int(comp_df['year'].max())}; {len(comp_df)} observations. {comp_status}")

if print_mode:
    st.markdown("---")
    st.markdown("### Printer-friendly report")
    st.write(f"**{country} · {INDICATOR_LABEL} · {UNITS}**")
    st.write(f"Coverage: {model['min_year']}–{model['max_year']} · n={model['n']} · Degree={degree}")
    st.write(f"Equation: {equation_text(model)}")
    st.write(f"In-sample R²={model['r2']:.5f}; in-sample MAE={model['mae']:.4f} years.")
    st.caption("Print mode simplifies the report presentation; it does not change the model or data.")
fig = make_chart(data, model, country, comparisons, sample_increment, horizon, print_mode=print_mode)
st.pyplot(fig, use_container_width=True)

st.subheader("Single-year estimate")
single_year = st.number_input("Year to estimate (1950–2100)", min_value=1950, max_value=2100, value=min(max(model["max_year"] + 1, 1950), 2100), step=1, key="single_year")
if st.button("Calculate value", key="calculate_single"):
    year = int(single_year)
    estimate = float(predict(model, [year])[0])
    label = classify_year(year, model, set(data["year"].astype(int)))
    observed = data.loc[data["year"] == year, "value"]
    st.session_state.single_result = {
        "year": year, "estimate": estimate, "label": label,
        "observed": float(observed.iloc[0]) if not observed.empty else None
    }
if "single_result" in st.session_state:
    r = st.session_state.single_result
    if r["year"] < model["min_year"] or r["year"] > model["max_year"]:
        st.warning("This is an extrapolation. Polynomial models can produce implausible results far outside the fitted range.")
    st.write(f"**{r['year']} estimate:** {r['estimate']:.4f} {UNITS} — {r['label']}.")
    if r["observed"] is not None:
        st.write(f"Observed source/working value for {r['year']}: **{r['observed']:.4f} years**. The model estimate is a fitted value and is not the same thing as the observation.")

st.subheader("Batch calculations")
batch_mode = st.radio("Batch input type", ["Year range", "Custom years"], horizontal=True)
valid_years = []
invalid_entries = []
if batch_mode == "Year range":
    c1, c2, c3 = st.columns(3)
    with c1:
        range_start = st.number_input("Start year", 1950, 2100, max(1950, model["min_year"]), 1, key="range_start")
    with c2:
        range_end = st.number_input("End year", 1950, 2100, min(2100, max(model["min_year"], model["max_year"])), 1, key="range_end")
    with c3:
        range_step = st.number_input("Step (1–10)", 1, 10, 1, 1, key="range_step")
    if range_end <= range_start:
        invalid_entries.append("End year must be greater than start year.")
    else:
        valid_years = list(range(int(range_start), int(range_end) + 1, int(range_step)))
        if not valid_years:
            invalid_entries.append("No years were generated.")
else:
    custom = st.text_input("Comma-separated years", value=f"{model['min_year']},{model['max_year']},{min(2100, model['max_year'] + 5)}")
    seen = set()
    for part in custom.split(","):
        part = part.strip()
        if not part:
            continue
        if not re.fullmatch(r"[+-]?\d+", part):
            invalid_entries.append(f"'{part}' is not an integer.")
            continue
        year = int(part)
        if year < 1950 or year > 2100:
            invalid_entries.append(f"{year} is outside 1950–2100.")
            continue
        if year in seen:
            invalid_entries.append(f"Duplicate year {year} ignored.")
            continue
        seen.add(year)
        valid_years.append(year)
if valid_years:
    st.write("Valid requested years:", ", ".join(map(str, valid_years)))
if invalid_entries:
    st.warning("Input feedback: " + " ".join(invalid_entries))
if st.button("Calculate batch", disabled=(len(valid_years) == 0), key="calculate_batch"):
    estimates = predict(model, valid_years)
    observed_years = set(data["year"].astype(int))
    rows = []
    for year, value in zip(valid_years, estimates):
        rows.append({
            "year": int(year), "estimate": float(value), "units": UNITS,
            "classification": classify_year(int(year), model, observed_years),
            "validated prediction bounds": "Not provided"
        })
    st.session_state.calc_results = pd.DataFrame(rows)
if isinstance(st.session_state.get("calc_results"), pd.DataFrame) and not st.session_state.calc_results.empty:
    st.dataframe(st.session_state.calc_results, use_container_width=True)
    for _, rr in st.session_state.calc_results.iterrows():
        if rr["year"] < model["min_year"] or rr["year"] > model["max_year"]:
            if rr["estimate"] < 0 or rr["estimate"] > 120:
                st.warning(f"Model estimate for {int(rr['year'])} is {rr['estimate']:.3g} years, which is implausible for life expectancy. It is shown unmodified rather than silently clipped.")

st.subheader("Calculus pathway: derivatives and turning points")
t_sym = None
try:
    import sympy as sp
    t_sym = sp.symbols("t")
    poly_expr = sum(float(c) * t_sym ** (len(model["coeff"]) - 1 - i) for i, c in enumerate(model["coeff"]))
    first_derivative = sp.diff(poly_expr, t_sym)
    second_derivative = sp.diff(first_derivative, t_sym)
    st.latex("y(t) = " + sp.latex(poly_expr))
    st.latex("y'(t) = " + sp.latex(first_derivative))
    st.latex("y''(t) = " + sp.latex(second_derivative))
    st.caption("Derivative units: y′ is years of life expectancy per calendar year; y″ is the change in that rate per calendar year squared. These describe the fitted polynomial, not directly observed annual rates.")
    critical = []
    try:
        roots = np.roots(np.polyder(model["coeff"]))
        for root in roots:
            if abs(root.imag) < 1e-7:
                year = float(root.real + model["base_year"])
                if model["min_year"] <= year <= model["max_year"]:
                    yval = float(predict(model, [year])[0])
                    second = float(np.polyval(np.polyder(model["coeff"], 2), root.real))
                    kind = "local minimum" if second > 1e-7 else ("local maximum" if second < -1e-7 else "inconclusive/flat")
                    critical.append({"year": round(year, 3), "fitted value (years)": round(yval, 4), "classification from second derivative": kind})
    except Exception:
        pass
    if critical:
        st.write("Critical points within the historical fitted interval:")
        st.dataframe(pd.DataFrame(critical), use_container_width=True)
    else:
        st.write("No real critical points were identified inside the historical fitted interval.")
    inflections = []
    try:
        roots2 = np.roots(np.polyder(model["coeff"], 2))
        for root in roots2:
            if abs(root.imag) < 1e-7:
                year = float(root.real + model["base_year"])
                if model["min_year"] <= year <= model["max_year"]:
                    inflections.append(round(year, 3))
    except Exception:
        pass
    st.write("Candidate inflection years inside the fitted interval (verify a concavity sign change):",
             ", ".join(map(str, inflections)) if inflections else "None identified.")
    st.caption("Historical-interval extrema must also compare both endpoints with all verified critical points. Inflection points require a concavity sign change; numerical candidates are not automatically proof.")
except ImportError:
    st.warning("SymPy is unavailable. Install the dependencies in requirements.txt to display derivative expressions.")

st.subheader("Mathematical interpretation and limitations")
st.markdown(
    f"""
- **Model type:** Degree-{degree} polynomial fitted to {model['n']} valid observations using centered time.
- **In-sample fit:** R² = {model['r2']:.5f}; MAE = {model['mae']:.4f} years. These describe fit to the same data used to train the model.
- **Uncertainty:** Residuals/MAE describe typical in-sample deviations but do not quantify all future uncertainty. No statistically validated 95% prediction intervals are provided.
- **Interpretation:** Life expectancy is a population-level period indicator. It is not a guarantee about any individual's lifespan, and the data alone do not establish why a trend occurred.
- **Limitations:** Annual values may be modeled/interpolated from life tables; source definitions and revisions can change. A polynomial can behave implausibly outside its fitted range.
- **Heritage connection:** Use sourced historical and cultural context to interpret the lives and policies behind the trend. Do not treat a single indicator as a ranking of a country's worth or identity.
"""
)
st.write("**Two average rates of change (model-based):**")
rate_years = st.columns(2)
with rate_years[0]:
    rate_start = st.number_input("Rate start year", 1950, 2100, model["min_year"], 1, key="rate_start")
with rate_years[1]:
    rate_end = st.number_input("Rate end year", 1950, 2100, model["max_year"], 1, key="rate_end")
if rate_end != rate_start:
    rate = (float(predict(model, [rate_end])[0]) - float(predict(model, [rate_start])[0])) / (rate_end - rate_start)
    st.write(f"Average modeled rate from {int(rate_start)} to {int(rate_end)}: **{rate:.5f} life-expectancy years per calendar year**.")
else:
    st.caption("Choose two different years to calculate an average rate of change.")

st.subheader("Degree comparison")
degree_compare = st.multiselect("Compare polynomial degrees", [3, 4, 5, 6, 7, 8], default=[3, 5])
degree_rows = []
for d in degree_compare:
    try:
        dm = fit_polynomial(data, d)
        degree_rows.append({"degree": d, "base year": dm["base_year"], "R² (in-sample)": dm["r2"], "MAE (years, in-sample)": dm["mae"], "equation": equation_text(dm, 5)})
    except ValueError as exc:
        degree_rows.append({"degree": d, "base year": "—", "R² (in-sample)": "Unavailable", "MAE (years, in-sample)": str(exc), "equation": "Unavailable"})
if degree_rows:
    st.dataframe(pd.DataFrame(degree_rows), use_container_width=True)
st.caption("A higher-degree polynomial can increase in-sample fit while becoming less stable or less plausible outside the observed years.")

st.subheader("Chronological holdout check")
if len(data) >= max(10, degree + 4):
    ordered = data.sort_values("year").reset_index(drop=True)
    holdout_n = max(3, int(math.ceil(len(ordered) * 0.2)))
    train, holdout = ordered.iloc[:-holdout_n], ordered.iloc[-holdout_n:]
    if len(train) >= degree + 1:
        hold_model = fit_polynomial(train, degree)
        hold_pred = predict(hold_model, holdout["year"].to_numpy())
        hold_mae = float(mean_absolute_error(holdout["value"], hold_pred))
        holdout_table = holdout.assign(predicted_value=hold_pred, absolute_error=np.abs(holdout["value"].to_numpy() - hold_pred))
        st.write(f"Chronological holdout MAE on {len(holdout)} latest observation(s): **{hold_mae:.4f} years**.")
        st.dataframe(holdout_table, use_container_width=True)
        st.caption("The model was fitted only on earlier years for this check. This small holdout is not proof of dependable long-range forecasting.")
    else:
        st.warning("Not enough earlier observations remain to fit the selected degree and reserve a holdout.")
else:
    st.warning(f"Insufficient observations for the configured holdout rule (n={len(data)}). No tested forecast skill is claimed.")

residual_frame = data.copy()
residual_frame["fitted_value"] = model["pred"]
residual_frame["residual"] = model["residuals"]

summary = {
    "Country": country,
    "Indicator": INDICATOR_LABEL,
    "Indicator code": WB_INDICATOR,
    "Units": UNITS,
    "Source": DATA_URL,
    "Retrieved (UTC)": retrieved_at,
    "Coverage": f"{model['min_year']}–{model['max_year']}",
    "Observations": model["n"],
    "Degree": degree,
    "Base year": model["base_year"],
    "Equation": equation_text(model),
    "R-squared (in-sample)": round(model["r2"], 7),
    "MAE (years)": round(model["mae"], 7),
    "Mean (years)": round(model["mean"], 7),
    "Data status": data_status,
}
findings = [
    f"The selected series contains {len(data)} valid observations from {model['min_year']} through {model['max_year']}.",
    f"The degree-{degree} polynomial has in-sample MAE {model['mae']:.4f} years; this is a descriptive fit measure.",
    "Country comparisons can only be interpreted cautiously because series coverage, revisions, and underlying estimates may differ."
]
results = st.session_state.calc_results if isinstance(st.session_state.get("calc_results"), pd.DataFrame) else pd.DataFrame()

st.subheader("Downloads")
csv_bytes = build_csv(data, country, edited)
xlsx_bytes = build_xlsx(data, summary, results, residual_frame, comparisons, edited)
try:
    pdf_bytes = build_pdf(data, summary, results, fig, findings, edited)
except Exception as exc:
    pdf_bytes = None
    st.warning(f"PDF generation is currently unavailable: {exc}")
d1, d2, d3 = st.columns(3)
with d1:
    st.download_button("Download CSV data", csv_bytes, file_name=f"{country.lower()}_life_expectancy_data.csv", mime="text/csv", use_container_width=True)
with d2:
    st.download_button("Download Excel workbook", xlsx_bytes, file_name=f"{country.lower()}_life_expectancy_report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
with d3:
    if pdf_bytes:
        st.download_button("Download PDF report", pdf_bytes, file_name=f"{country.lower()}_life_expectancy_report.pdf", mime="application/pdf", use_container_width=True)
    else:
        st.button("PDF unavailable", disabled=True, use_container_width=True)

st.markdown("---")
st.subheader("Research and heritage evidence starter")
st.markdown(
    """
**Suggested research question:** How did life expectancy at birth change in Mexico and Argentina, and how well do polynomial models describe the historical patterns without overstating future certainty?

**Primary data source:** World Bank, *Life expectancy at birth, total (years)*, indicator `SP.DYN.LE00.IN`.

**Context sources to consult and cite in your written brief:**
1. World Bank indicator page and metadata (definition, source notes, data revisions).
2. United Nations World Population Prospects, for demographic methodology and context.
3. A reputable country-specific historical or public-health source from a national statistics office, university, WHO/PAHO, or peer-reviewed publication.

Do not invent a historical cause from the graph alone. Add a specific sourced example of a public-health achievement, community experience, policy, or historical change before making a heritage connection.
"""
)
st.caption("This app retrieves data at runtime. The project package does not contain a fabricated or fallback data series.")
