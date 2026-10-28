# Hispanic Heritage Historical Data Explorer

A Streamlit research app for investigating historical life expectancy at birth using genuine World Bank data and polynomial models. The featured comparison is Mexico and Argentina. The app also offers the other countries specified by the assignment.

## Research question

How has life expectancy at birth changed over time in Mexico and Argentina, and what can polynomial models reveal—and fail to reveal—about these trends?

## Files

- `app.py` — complete Streamlit application.
- `requirements.txt` — Python dependencies.
- `sources.md` — source definitions, citations, and research context starter.
- `tests.md` — manual acceptance-test checklist.
- `proposal.md` — editable research proposal draft.
- `math_brief_template.md` — brief structure and calculations to complete from app results.

## Run locally

Python 3.10 or later is recommended.

```bash
python -m venv .venv
```

Activate the environment:

- Windows PowerShell: `.venv\Scripts\Activate.ps1`
- macOS/Linux: `source .venv/bin/activate`

Then run:

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

The app needs an internet connection to retrieve live World Bank data. It reports API errors and never substitutes invented data.

## Features

- Required nine-country menu: Mexico, Brazil, Argentina, Colombia, Chile, Peru, Venezuela, Ecuador, Uruguay.
- Official World Bank indicator `SP.DYN.LE00.IN`.
- Target window: current year minus 70 through current year minus 1.
- Editable year/value table, validation, and reset to source.
- Centered-time polynomial fit, degree 3–8, in-sample R² and MAE, residuals.
- Same-indicator country comparisons, sample increment, forecast display horizon, and printer-friendly view.
- Single-year and batch calculations for years 1950–2100.
- Calculus pathway: first and second derivative expressions, candidate critical points, and candidate inflection years.
- Chronological holdout evaluation when enough data are available.
- Genuine CSV, XLSX, and PDF downloads.

## Data and modeling limitations

- Life expectancy at birth is a period measure based on mortality patterns, not a prediction of an individual's lifespan.
- World Bank data can be revised and some annual series are estimated/interpolated.
- In-sample metrics describe the fit to the fitted data; they do not establish forecasting accuracy.
- A chronological holdout provides a limited check, not proof of future accuracy.
- No statistically validated prediction intervals are supplied.
- Polynomial extrapolation can be implausible; displayed values are not silently clipped.
- Numerical roots for critical/inflection points should be checked, and extrema on the historical interval must include both endpoints.

## Deployment

### Streamlit Community Cloud

1. Create a GitHub repository and upload these files.
2. Sign in to Streamlit Community Cloud and connect the repository.
3. Create an app using `app.py` as the entry point.
4. Deploy and inspect logs if startup fails.
5. Open the public URL in a fresh session and test live data and all downloads.

### Render

Create a Python Web Service connected to the repository.

Build command:

```bash
pip install -r requirements.txt
```

Start command (one line):

```bash
python -m streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT
```

## AI-use and student responsibility

AI assisted with drafting this starter implementation. The student team must run the app, verify official source values, inspect formulas and exports, document errors/fixes, add context research, and be able to explain each member's contribution. Do not claim the app is tested or deployed until you have actually completed those checks.
