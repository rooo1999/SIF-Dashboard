"""
Fund NAV Performance Dashboard
--------------------------------
Upload (or auto-load) an Excel file of daily NAVs for multiple funds plus
Nifty 50 / Nifty 500 benchmarks. Pick a start and end date, and the app:

1. Keeps only funds that have real (non-NA) NAVs across the whole window.
2. Rebases every series to 100 at the start date so they're comparable
   even if funds were launched at different NAVs / dates.
3. Plots the indexed "growth of 100" journey.
4. Shows a table of indexed values.
5. Shows a trailing-returns table (1D, 1W, 1M, 3M, 6M, 1Y, 3Y, 5Y, Since
   Inception) computed from the FULL history of each fund (not just the
   selected window), so returns are always as-of the selected end date.

Run with:  streamlit run fund_dashboard.py
"""

import io
import datetime as dt

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Fund NAV Performance Dashboard", layout="wide")

DEFAULT_FILE = "All_Hybrid_SIF_NAVs.xlsx"

BENCHMARK_NAMES = {"Nifty 50", "Nifty 500", "NIFTY 50", "NIFTY 500"}


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data(file_bytes: bytes) -> pd.DataFrame:
    """Load the NAV excel file. First column = Date, rest = fund/benchmark NAVs.
    Non-numeric values (e.g. 'NA', '#N/A', blanks) become NaN (fund not live)."""
    df = pd.read_excel(io.BytesIO(file_bytes))
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.rename(columns={date_col: "Date"})

    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("Date").reset_index(drop=True)
    return df


def get_series_names(df: pd.DataFrame):
    all_cols = [c for c in df.columns if c != "Date"]
    benchmarks = [c for c in all_cols if c in BENCHMARK_NAMES]
    funds = [c for c in all_cols if c not in BENCHMARK_NAMES]
    return funds, benchmarks


# --------------------------------------------------------------------------
# Core calculations
# --------------------------------------------------------------------------
def snap_to_previous_trading_day(df: pd.DataFrame, target: dt.date) -> pd.Timestamp:
    """If `target` isn't a date present in the sheet (weekend/holiday/gap),
    snap back to the most recent available date on or before it. If target
    is earlier than every date in the sheet, fall back to the earliest
    available date."""
    ts = pd.Timestamp(target)
    valid = df["Date"][df["Date"] <= ts]
    if valid.empty:
        return df["Date"].min()
    return valid.max()


def filter_available_in_window(df: pd.DataFrame, start: dt.date, end: dt.date, cols):
    """A series is 'available' for the selected window only if it has a
    valid (non-NaN) NAV on every trading day the sheet has between start
    and end (i.e. it was already listed for the whole window). Start/end
    dates that aren't in the sheet are snapped back to the previous
    available trading day."""
    start_ts = snap_to_previous_trading_day(df, start)
    end_ts = snap_to_previous_trading_day(df, end)
    window = df[(df["Date"] >= start_ts) & (df["Date"] <= end_ts)]
    if window.empty:
        return [], window, start_ts, end_ts
    available = [c for c in cols if window[c].notna().all()]
    return available, window, start_ts, end_ts


def build_indexed(window: pd.DataFrame, cols, base=100.0):
    """Rebase each column so its first value in the window = base."""
    idx_df = window[["Date"] + cols].copy()
    for c in cols:
        first_val = idx_df[c].iloc[0]
        idx_df[c] = idx_df[c] / first_val * base
    return idx_df


def trailing_return(full_df: pd.DataFrame, col: str, as_of: pd.Timestamp, days_back, kind="calendar"):
    """Compute trailing return for a column as of `as_of`, looking back
    `days_back` calendar days (kind='calendar') or using the first row
    (kind='inception'). Returns None if data isn't available that far back."""
    series = full_df[["Date", col]].dropna()
    if series.empty:
        return None

    series = series[series["Date"] <= as_of]
    if series.empty:
        return None

    end_val = series[col].iloc[-1]
    end_date = series["Date"].iloc[-1]

    if kind == "inception":
        start_val = series[col].iloc[0]
        start_date = series["Date"].iloc[0]
        if start_date == end_date:
            return None
        years = (end_date - start_date).days / 365.25
        if years <= 0:
            return None
        if years < 1:
            return (end_val / start_val - 1) * 100
        cagr = ((end_val / start_val) ** (1 / years) - 1) * 100
        return cagr

    target_date = as_of - pd.Timedelta(days=days_back)
    prior = series[series["Date"] <= target_date]
    if prior.empty:
        return None
    start_val = prior[col].iloc[-1]
    start_date = prior["Date"].iloc[-1]

    years = (end_date - start_date).days / 365.25
    if days_back >= 365 and years >= 1:
        # annualize for periods of 1yr or more
        if start_val <= 0:
            return None
        return ((end_val / start_val) ** (1 / years) - 1) * 100
    if start_val == 0:
        return None
    return (end_val / start_val - 1) * 100


PERIODS = [
    ("1D", 1),
    ("1W", 7),
    ("1M", 30),
    ("3M", 91),
    ("6M", 182),
    ("1Y", 365),
    ("3Y", 365 * 3),
    ("5Y", 365 * 5),
]


def build_trailing_table(full_df: pd.DataFrame, cols, as_of: pd.Timestamp):
    rows = []
    for c in cols:
        row = {"Fund": c}
        for label, days in PERIODS:
            row[label] = trailing_return(full_df, c, as_of, days)
        row["Since Inception (CAGR/Abs)"] = trailing_return(full_df, c, as_of, None, kind="inception")
        rows.append(row)
    table = pd.DataFrame(rows).set_index("Fund")
    return table


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
st.title("📈 Fund NAV Performance Dashboard")
st.caption(
    "Compare fund performance over any date range, rebased to a common "
    "starting point of 100, against Nifty 50 and Nifty 500."
)

with st.sidebar:
    st.header("Data source")
    data_source = st.radio(
        "Where should NAVs come from?",
        ["Auto-fetch (API + yfinance)", "Upload Excel (manual)"],
        index=0,
    )

if data_source == "Auto-fetch (API + yfinance)":
    import upvaly_client as uc
    import data_pipeline as dp

    with st.sidebar:
        st.header("Funds")
        refresh_meta = st.button("🔄 Refresh fund list from API")

        with st.spinner("Loading fund list & categories..."):
            meta = dp.load_all_metadata(force_refresh=refresh_meta)

        meta_errors = {n: m for n, m in meta.items() if m.get("error")}
        categories = sorted({m.get("category", "Unknown") for m in meta.values()})

        category_filter = st.multiselect("Filter by category", categories)
        fund_filter = st.multiselect("Or choose specific fund(s)", uc.FUND_LIST)

        st.caption("Leave both empty to include every tracked fund. Selecting either narrows the set; selecting both combines them.")

        with st.expander("🐞 Debug: raw API response"):
            debug_fund = st.selectbox("Fund to inspect", uc.FUND_LIST)
            if st.button("Fetch raw JSON"):
                try:
                    raw = uc.debug_fetch_scheme(debug_fund)
                    st.json(raw)
                except Exception as e:  # noqa: BLE001
                    st.error(f"Request failed: {e}")

    if category_filter or fund_filter:
        selected_names = set()
        if category_filter:
            selected_names |= {n for n, m in meta.items() if m.get("category", "Unknown") in category_filter}
        if fund_filter:
            selected_names |= set(fund_filter)
        selected_names = [n for n in uc.FUND_LIST if n in selected_names]
    else:
        selected_names = uc.FUND_LIST

    if meta_errors:
        st.sidebar.warning(f"⚠️ Couldn't fetch metadata for {len(meta_errors)} fund(s) — see Debug panel.")

    with st.spinner(f"Fetching NAV history for {len(selected_names)} fund(s) + benchmarks..."):
        df = dp.build_wide_dataframe(selected_names, meta)
        failed = dp.fetch_failures(selected_names, meta)

    if df.empty or "Date" not in df.columns or df.shape[1] <= 1:
        st.error(
            "Couldn't build any NAV series from the API. Open the Debug panel in the "
            "sidebar, fetch a raw response, and check it matches the expected shape."
        )
        st.stop()

    if failed:
        st.sidebar.caption(f"⚪ No scheme code found for: {', '.join(failed)}")

else:
    with st.sidebar:
        st.header("Data")
        uploaded = st.file_uploader("Upload NAV Excel file", type=["xlsx", "xls"])

    if uploaded is not None:
        file_bytes = uploaded.read()
    else:
        try:
            with open(DEFAULT_FILE, "rb") as f:
                file_bytes = f.read()
            st.sidebar.info(f"Using bundled file: {DEFAULT_FILE}")
        except FileNotFoundError:
            st.warning("Upload a NAV Excel file to get started (Date column + fund/benchmark NAV columns).")
            st.stop()

    df = load_data(file_bytes)

funds, benchmarks = get_series_names(df)
all_cols = funds + benchmarks

if df["Date"].isna().all() or df.empty:
    st.error("No usable dates in the data.")
    st.stop()

min_date = df["Date"].min().date()
max_date = df["Date"].max().date()

with st.sidebar:
    st.header("Date Range")
    start_date = st.date_input("Start date", value=min_date, min_value=min_date, max_value=max_date)
    end_date = st.date_input("End date", value=max_date, min_value=min_date, max_value=max_date)

    st.header("Series")
    show_benchmarks = st.multiselect("Benchmarks", benchmarks, default=benchmarks)

if start_date >= end_date:
    st.error("Start date must be before end date.")
    st.stop()

# Only include funds available for the entire window; benchmarks are
# always assumed available (they existed long before any fund).
available_funds, window, actual_start, actual_end = filter_available_in_window(
    df, start_date, end_date, funds
)
available_benchmarks = [b for b in show_benchmarks if window[b].notna().all()] if not window.empty else []

excluded_funds = [f for f in funds if f not in available_funds]

plot_cols = available_funds + available_benchmarks

if not plot_cols:
    st.error("No funds or benchmarks have complete data for this date range. Try a shorter or more recent window.")
    st.stop()

st.subheader(f"Performance: {actual_start.date()} → {actual_end.date()}")

if actual_start.date() != start_date or actual_end.date() != end_date:
    st.caption(
        f"ℹ️ {start_date} / {end_date} aren't trading days in the sheet — "
        f"snapped back to the previous available date(s): "
        f"{actual_start.date()} / {actual_end.date()}."
    )

if excluded_funds:
    st.caption(
        f"⚪ Excluded (not yet listed for full window): {', '.join(excluded_funds)}"
    )

# --- Indexed journey ---
indexed = build_indexed(window, plot_cols, base=100.0)

fig = go.Figure()
for c in plot_cols:
    is_bench = c in benchmarks
    fig.add_trace(
        go.Scatter(
            x=indexed["Date"],
            y=indexed[c],
            mode="lines",
            name=c,
            line=dict(width=3 if is_bench else 2, dash="dash" if is_bench else "solid"),
        )
    )
fig.update_layout(
    height=550,
    hovermode="x unified",
    yaxis_title="Indexed Value (Start = 100)",
    xaxis_title="Date",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(t=30, b=10),
)
st.plotly_chart(fig, use_container_width=True)

# --- Summary of period return (from indexed chart) ---
st.subheader("Summary — Selected Period Return")
period_summary = pd.DataFrame({
    "Start NAV": window[plot_cols].iloc[0],
    "End NAV": window[plot_cols].iloc[-1],
    "Indexed End (Start=100)": indexed[plot_cols].iloc[-1],
    "Period Return (%)": (indexed[plot_cols].iloc[-1] - 100),
}).round(2)
period_summary.index.name = "Fund"
st.dataframe(period_summary.sort_values("Period Return (%)", ascending=False), use_container_width=True)

# --- Indexed values table ---
st.subheader("Indexed NAV Table (Start = 100)")
display_indexed = indexed.copy()
display_indexed["Date"] = display_indexed["Date"].dt.strftime("%Y-%m-%d")
st.dataframe(display_indexed.round(2), use_container_width=True, height=350)

# --- Trailing returns table (uses FULL history, as-of end_date) ---
st.subheader("Trailing Returns")
st.caption(
    "Computed on each fund's/benchmark's full available history as of the "
    "selected end date — not limited to the chart window. Periods ≥ 1Y are "
    "annualized (CAGR); shorter periods are absolute returns. Blank = not "
    "enough history yet for that period."
)
trailing_cols = all_cols  # show all funds+benchmarks regardless of window availability
trailing_table = build_trailing_table(df, trailing_cols, actual_end)
st.dataframe(trailing_table.round(2), use_container_width=True, height=400)

st.divider()
with st.expander("ℹ️ How this works"):
    st.markdown(
        """
- **Availability filter**: a fund is only plotted for a chosen date range if it has a real NAV
  (not NA/blank) on *every* day in that range — i.e. it was already listed for the whole window.
- **Rebasing to 100**: every included fund's first NAV in the selected window is set to 100,
  and all subsequent values are scaled proportionally — this makes funds with very different
  NAV levels (e.g. ₹10 vs ₹40,000 for an index) directly comparable on one chart.
- **Trailing returns**: always computed on the full uploaded history (independent of the
  chart's date range), as of the selected end date, so a fund that's only a few months old
  will simply show blanks for 3Y/5Y until enough history accumulates.
- **New fund launches**: as new funds get listed in future updates of the sheet, they'll
  automatically show up here — no code changes needed.
- **Auto-fetch mode**: fund NAVs come from the finapi.upvaly.com API (scheme lookup +
  full NAV history), Nifty 50 / Nifty 500 come from yfinance. Each fund/benchmark's
  history is cached to disk after the first fetch, so later runs only pull the new days
  since your last run instead of re-downloading everything.
- **Category & fund filters**: leave both sidebar boxes empty to see every tracked fund;
  picking a category includes all funds in it, picking specific funds adds those too —
  the two combine rather than override each other.
"""
    )
