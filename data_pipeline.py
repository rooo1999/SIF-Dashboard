"""Combine upvaly fund NAV data + yfinance benchmark data into one wide
DataFrame: Date, <fund columns...>, Nifty 50, Nifty 500 — the same shape the
rest of the dashboard (indexing, trailing returns, etc.) already expects."""

import re

import pandas as pd

import upvaly_client as uc
import benchmark_client as bc

_CATEGORY_ABBR = [
    (r"Active Asset Allocator Long-?\s*Short", "AAA"),
    (r"Equity Ex-?\s*Top\s*100 Long\s*-?\s*Short", "Eq ExT100"),
    (r"Sector Rotation Long-?\s*Short", "SecRot"),
    (r"Equity Long\s*-?\s*Short", "Equity"),
    (r"Hybrid Long-?\s*Short", "Hybrid"),
]


def short_label(scheme_name: str) -> str:
    """Turn a long AMFI scheme name into a compact, still-distinguishing
    label, e.g. 'Arudha Hybrid Long-Short Fund-Regular Plan-Growth' ->
    'Arudha Hybrid'."""
    amc = scheme_name.split()[0].strip()
    # normalise casing for all-caps / all-lower AMC names
    amc = amc[0].upper() + amc[1:] if amc.isupper() or amc.islower() else amc
    suffix = None
    for pattern, abbr in _CATEGORY_ABBR:
        if re.search(pattern, scheme_name, flags=re.IGNORECASE):
            suffix = abbr
            break
    return f"{amc} {suffix}" if suffix else amc


def build_short_labels(scheme_names) -> dict:
    """Map scheme_name -> short_label, resolving collisions by falling back
    to the full name for any duplicates."""
    labels = {name: short_label(name) for name in scheme_names}
    seen = {}
    for name, lab in labels.items():
        seen.setdefault(lab, []).append(name)
    for lab, names in seen.items():
        if len(names) > 1:
            for name in names:
                labels[name] = name  # not unique — use full name instead
    return labels


def load_all_metadata(force_refresh=False, progress_cb=None):
    """Returns {scheme_name: {scheme_code, category, ...}} for the full
    tracked fund list."""
    return uc.get_all_scheme_meta(uc.FUND_LIST, force_refresh=force_refresh, progress_cb=progress_cb)


def build_wide_dataframe(selected_fund_names, meta: dict, include_benchmarks=("Nifty 50", "Nifty 500")):
    """Fetches NAV history for each selected fund + the requested
    benchmarks and assembles the wide Date-indexed DataFrame."""
    labels = build_short_labels(selected_fund_names)

    series_frames = []
    for name in selected_fund_names:
        info = meta.get(name, {})
        code = info.get("scheme_code")
        if not code:
            continue
        hist = uc.get_full_nav_history(code, scheme_name=name)
        if hist.empty:
            continue
        col = labels[name]
        s = hist.set_index("date")["nav"].rename(col)
        series_frames.append(s)

    for label in include_benchmarks:
        hist = bc.get_benchmark_history(label)
        if hist.empty:
            continue
        s = hist.set_index("date")["nav"].rename(label)
        series_frames.append(s)

    if not series_frames:
        return pd.DataFrame(columns=["Date"])

    wide = pd.concat(series_frames, axis=1).sort_index()
    wide.index.name = "Date"
    wide = wide.reset_index()
    return wide


def fetch_failures(selected_fund_names, meta: dict):
    """Names for which we have no usable scheme_code, so the caller can
    surface a warning instead of silently dropping them."""
    return [n for n in selected_fund_names if not meta.get(n, {}).get("scheme_code")]
