"""
Client for the finapi.upvaly.com mutual-fund / SIF API.

NOTE ON RELIABILITY: this was written without live access to the API (the
sandbox this was built in can't reach finapi.upvaly.com), so the exact JSON
field names are an educated guess based on common conventions for this kind
of API. Every parser below tries several likely key names before giving up,
and `debug_fetch_scheme()` / the "Debug: raw API response" panel in the
Streamlit app let you see the actual JSON. If a fund's category or NAV
history isn't parsing correctly, grab that raw JSON and it's a one-line fix.
"""

import time
import json
import datetime as dt
from pathlib import Path
from urllib.parse import quote

import requests
import pandas as pd

BASE_URL = "https://finapi.upvaly.com"
CACHE_DIR = Path(__file__).parent / "nav_cache"
CACHE_DIR.mkdir(exist_ok=True)
META_CACHE_FILE = CACHE_DIR / "scheme_meta.json"

# The full list of SIF schemes to track.
FUND_LIST = [
    "Altiva Equity Ex- Top 100 Long - Short Fund - Regular Plan - Growth",
    "Altiva Hybrid Long-Short Fund - Regular Plan - Growth",
    "Apex Hybrid Long-Short Fund - Regular - Growth",
    "Arthaya Equity Long Short Fund - Regular Plan - Growth Option",
    "Arudha Equity Long-Short Fund-Regular Plan-Growth",
    "Arudha Hybrid Long-Short Fund-Regular Plan-Growth",
    "Diviniti Equity Long Short Fund - Regular Plan Growth Option",
    "DynaSIF Active Asset Allocator Long-Short Fund - Regular Plan - Growth Option",
    "DynaSIF Equity Ex-Top 100 Long - Short Fund - Regular Plan - Growth Option",
    "DynaSIF Equity Long - Short Fund - Regular Plan - Growth Option",
    "Sapphire Equity Long-Short SIF - Growth",
    "RedHex Hybrid Long-Short Fund - Regular - Growth",
    "Summit Equity Long-Short Fund - Regular Plan - Growth",
    "iSIF Active Asset Allocator Long-Short Fund - Growth",
    "iSIF Equity Ex-Top 100 Long-Short Fund - Growth",
    "iSIF Equity Long-Short Fund - Growth",
    "iSIF Hybrid Long-Short Fund - Growth",
    "Prism Hybrid Long-Short Fund - Regular Plan- Growth Option",
    "INFINITY HYBRID LONG-SHORT FUND-REGULAR - GROWTH",
    "Magnum Hybrid Long Short Fund - Regular Plan - Growth",
    "Platinum Hybrid Long-Short Fund - Regular Plan - Growth",
    "qsif Active Asset Allocator Long-Short Fund - Growth Option - Regular Plan",
    "qsif Equity Ex-Top 100 Long-Short Fund - Growth Option - Regular Plan",
    "qsif Equity Long Short Fund - Growth Option - Regular Plan",
    "qsif Hybrid Long-Short Fund - Growth Option - Regular Plan",
    "qsif Sector Rotation Long-Short Fund - Growth Option - RegularPlan",
    "WSIF Equity Ex-Top 100 Long-Short Fund - Regular Growth",
    "WSIF Equity Long-Short Fund - Regular Growth",
    "Titanium Equity Long-Short Fund Regular Growth",
    "Titanium Hybrid Long-Short Fund Regular Plan Growth",
]

# Earliest date to ask the API for when we have no cache yet. These are all
# newly launched SIFs (~late 2025), but this is set conservatively low in
# case older/other schemes get added later.
FLOOR_DATE = dt.date(2020, 1, 1)

_session = requests.Session()
_session.headers.update({"User-Agent": "Mozilla/5.0 (fund-dashboard/1.0)"})


def _get(url, params=None, retries=3, timeout=15):
    last_err = None
    for attempt in range(retries):
        try:
            resp = _session.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(0.6 * (attempt + 1))
    raise RuntimeError(f"GET {url} failed after {retries} attempts: {last_err}")


# --------------------------------------------------------------------------
# Flexible parsing helpers — try several likely key names.
# --------------------------------------------------------------------------
def _first_key(d: dict, keys, default=None):
    for k in keys:
        if isinstance(d, dict) and k in d and d[k] is not None:
            return d[k]
    return default


def _unwrap(obj, wrapper_keys):
    """If obj is a dict wrapping the real payload (e.g. {'data': {...}}),
    unwrap it. Otherwise return obj unchanged."""
    if isinstance(obj, dict):
        for k in wrapper_keys:
            if k in obj:
                return obj[k]
    return obj


CODE_KEYS = ["schemeCode", "scheme_code", "code", "id", "schemeId"]
CATEGORY_KEYS = [
    "category", "schemeCategory", "scheme_category", "subCategory",
    "sub_category", "fundCategory", "categoryName", "class", "schemeClass",
    "type", "schemeType",
]
NAME_KEYS = ["schemeName", "scheme_name", "name"]
INCEPTION_KEYS = ["inceptionDate", "inception_date"]
FUND_HOUSE_KEYS = ["fundHouse", "fund_house", "companyName"]
NAV_LIST_WRAPPER_KEYS = ["navHistory", "nav_history", "history", "navs", "data", "result", "nav"]
NAV_DATE_KEYS = ["date", "navDate", "nav_date", "asOfDate", "as_of_date"]
NAV_VALUE_KEYS = ["nav", "navValue", "nav_value", "value", "netAssetValue"]


def parse_scheme_meta(raw: dict) -> dict:
    """Extract {scheme_code, scheme_name, category, inception_date,
    fund_house} from a scheme-name response. Response is wrapped as
    {"status": "success", "data": {...}} — an explicit non-success status
    is treated as an error."""
    if isinstance(raw, dict) and raw.get("status") not in (None, "success"):
        raise RuntimeError(f"API returned status={raw.get('status')!r}: {raw.get('message')}")
    payload = _unwrap(raw, ["data", "scheme", "result"])
    if isinstance(payload, list) and payload:
        payload = payload[0]
    code = _first_key(payload, CODE_KEYS)
    category = _first_key(payload, CATEGORY_KEYS, default="Unknown")
    name = _first_key(payload, NAME_KEYS)
    inception = _first_key(payload, INCEPTION_KEYS)
    fund_house = _first_key(payload, FUND_HOUSE_KEYS)
    return {
        "scheme_code": code, "scheme_name": name, "category": category,
        "inception_date": inception, "fund_house": fund_house, "raw": raw,
    }


def parse_nav_entries(raw) -> pd.DataFrame:
    """Extract a (date, nav) DataFrame from a NAV-history-shaped response.
    Handles both a bare list and the confirmed real shape
    {"status": "success", "data": {"navHistory": [{"navDate":..,"nav":..}]}}."""
    if isinstance(raw, dict) and raw.get("status") not in (None, "success"):
        return pd.DataFrame(columns=["date", "nav"])
    entries = _unwrap(raw, NAV_LIST_WRAPPER_KEYS)
    if isinstance(entries, dict):
        # sometimes a dict itself wraps one more level, e.g. {"navHistory": {"data": [...]}}
        entries = _unwrap(entries, NAV_LIST_WRAPPER_KEYS)
    if not isinstance(entries, list):
        return pd.DataFrame(columns=["date", "nav"])

    rows = []
    for e in entries:
        if not isinstance(e, dict):
            continue
        d = _first_key(e, NAV_DATE_KEYS)
        v = _first_key(e, NAV_VALUE_KEYS)
        if d is None or v is None:
            continue
        rows.append({"date": d, "nav": v})
    if not rows:
        return pd.DataFrame(columns=["date", "nav"])
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=False)
    df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
    df = df.dropna(subset=["date", "nav"]).drop_duplicates("date").sort_values("date")
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------
# Public fetch functions
# --------------------------------------------------------------------------
def fetch_scheme_meta_raw(scheme_name: str) -> dict:
    url = f"{BASE_URL}/api/mf/scheme-name/{quote(scheme_name, safe='')}"
    return _get(url)


def fetch_nav_range_raw(scheme_code, start: dt.date, end: dt.date) -> dict:
    url = f"{BASE_URL}/api/mf/scheme-code/{scheme_code}/nav"
    # Try the most common param naming; if the API uses different names,
    # this still sends *a* request — check the debug panel if results come
    # back empty.
    params = {
        "startDate": start.isoformat(),
        "endDate": end.isoformat(),
        "from": start.isoformat(),
        "to": end.isoformat(),
    }
    return _get(url, params=params)


def load_scheme_meta_cache() -> dict:
    if META_CACHE_FILE.exists():
        try:
            return json.loads(META_CACHE_FILE.read_text())
        except Exception:  # noqa: BLE001
            return {}
    return {}


def save_scheme_meta_cache(meta: dict):
    META_CACHE_FILE.write_text(json.dumps(meta, indent=2, default=str))


def get_all_scheme_meta(fund_list=None, force_refresh=False, progress_cb=None) -> dict:
    """Returns {scheme_name: {scheme_code, category, ...}} for every fund in
    fund_list, using a disk cache so we don't hit the API every run."""
    fund_list = fund_list or FUND_LIST
    meta = {} if force_refresh else load_scheme_meta_cache()
    changed = False
    for i, name in enumerate(fund_list):
        if name not in meta:
            try:
                raw = fetch_scheme_meta_raw(name)
                parsed = parse_scheme_meta(raw)
                parsed.pop("raw", None)
                meta[name] = parsed
                changed = True
            except Exception as e:  # noqa: BLE001
                meta[name] = {"scheme_code": None, "scheme_name": name, "category": "Unknown", "error": str(e)}
                changed = True
        if progress_cb:
            progress_cb(i + 1, len(fund_list), name)
    if changed:
        save_scheme_meta_cache(meta)
    return meta


def _fund_cache_file(scheme_code) -> Path:
    return CACHE_DIR / f"scheme_{scheme_code}.csv"


def get_full_nav_history(scheme_code, scheme_name="") -> pd.DataFrame:
    """Loads cached NAV history from disk and fetches only the missing
    (incremental) date range from the API, so repeated runs are cheap."""
    cache_file = _fund_cache_file(scheme_code)
    if cache_file.exists():
        cached = pd.read_csv(cache_file, parse_dates=["date"])
    else:
        cached = pd.DataFrame(columns=["date", "nav"])

    today = pd.Timestamp(dt.date.today())
    if cached.empty:
        fetch_start = FLOOR_DATE
    else:
        fetch_start = (cached["date"].max() + pd.Timedelta(days=1)).date()

    if fetch_start <= dt.date.today():
        try:
            raw = fetch_nav_range_raw(scheme_code, fetch_start, dt.date.today())
            new_df = parse_nav_entries(raw)
            if not new_df.empty:
                cached = (
                    pd.concat([cached, new_df], ignore_index=True)
                    .drop_duplicates("date")
                    .sort_values("date")
                    .reset_index(drop=True)
                )
                cached.to_csv(cache_file, index=False)
        except Exception:  # noqa: BLE001
            pass  # fall back to whatever's cached; UI surfaces staleness separately

    return cached


def debug_fetch_scheme(scheme_name: str):
    """Returns the raw JSON for one scheme-name lookup, for the debug panel."""
    return fetch_scheme_meta_raw(scheme_name)
