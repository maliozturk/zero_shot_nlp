#!/usr/bin/env python3
"""
geolife_enrich.py

End-to-end pipeline:
 - load .plt Geolife files
 - detect stays (dwell >= 20 min, radius <= 200 m)
 - query OSM (Overpass preferred, Nominatim fallback)
 - map tags -> category
 - dedupe POIs and output CSV

Usage:
  python geolife_enrich.py --input_dir path/to/plt_dir --output stays_enriched.csv
"""

import os
import sys
import time
import json
import math
import glob
import hashlib
import logging
import argparse
import tempfile
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Tuple, Optional

import pandas as pd
import numpy as np
import requests
from dateutil import parser as dateparser
from geopy.distance import geodesic  # pip install geopy
from shapely.geometry import Point  # pip install shapely
from tqdm import tqdm

# -----------------------
# CONFIG / DEFAULTS
# -----------------------
USER_AGENT = "GeolifePOIEnricher/1.0 (+https://example.org; contact: youremail@example.com)"
CONTACT_EMAIL = "youremail@example.com"
OVERPASS_ENDPOINT = "https://overpass-api.de/api/interpreter"
NOMINATIM_REVERSE = "https://nominatim.openstreetmap.org/reverse"
CACHE_DIR = os.path.expanduser("~/.geolife_cache")
MIN_REQUEST_INTERVAL = 1.0  # seconds, default throttle (1 req/sec)
MAX_RETRIES = 4
BACKOFF_FACTOR = 1.5

# Stay detection defaults
DWELL_SECONDS = 20 * 60       # 20 minutes
DIST_MAX_METERS = 200         # 200 m

# POI query defaults
QUERY_RADIUS_M = 200          # default radius for POI search (100-300 m recommended)
POI_DEDUPE_RADIUS_M = 50      # if POIs within 50 m, consider duplicate

# Output CSV columns
OUTPUT_COLUMNS = [
    "user_id", "arrive_time", "leave_time", "lat", "lon",
    "poi_name", "poi_category", "poi_lat", "poi_lon",
    "match_distance_m", "query_radius_m", "confidence_score", "source"
]

# Create cache dir
os.makedirs(CACHE_DIR, exist_ok=True)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


# -----------------------
# Utilities
# -----------------------
def _hash_request(key: str) -> str:
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def disk_cache_get(key: str) -> Optional[dict]:
    """Return cached JSON or None."""
    h = _hash_request(key)
    path = os.path.join(CACHE_DIR, f"{h}.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def disk_cache_set(key: str, value: dict):
    h = _hash_request(key)
    path = os.path.join(CACHE_DIR, f"{h}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(value, f)


_last_request_time = 0.0


def throttle():
    """Ensure at least MIN_REQUEST_INTERVAL between requests (global)."""
    global _last_request_time
    now = time.time()
    delta = now - _last_request_time
    if delta < MIN_REQUEST_INTERVAL:
        time.sleep(MIN_REQUEST_INTERVAL - delta)
    _last_request_time = time.time()


def safe_request(method: str, url: str, params=None, data=None, headers=None, use_cache=True) -> Optional[dict]:
    """
    Simple request wrapper with disk cache, throttle, and exponential backoff.
    Caches by URL + params + data.
    """
    headers = headers or {}
    key = json.dumps({"method": method, "url": url, "params": params or {}, "data": data or {}}, sort_keys=True)
    if use_cache:
        cached = disk_cache_get(key)
        if cached is not None:
            return cached
    retries = 0
    backoff = BACKOFF_FACTOR
    while retries <= MAX_RETRIES:
        throttle()
        try:
            if method.upper() == "GET":
                r = requests.get(url, params=params, headers=headers, timeout=30)
            else:
                r = requests.post(url, data=data, headers=headers, timeout=60)
        except requests.RequestException as e:
            logging.warning("Request error: %s (retrying...)", e)
            time.sleep(backoff)
            retries += 1
            backoff *= BACKOFF_FACTOR
            continue
        if r.status_code == 200:
            try:
                j = r.json()
            except ValueError:
                j = {"text": r.text}
            if use_cache:
                disk_cache_set(key, j)
            return j
        elif r.status_code in (429, 502, 503, 504, 500):
            logging.warning("HTTP %s from %s. Backoff and retry.", r.status_code, url)
            time.sleep(backoff)
            retries += 1
            backoff *= BACKOFF_FACTOR
            continue
        else:
            logging.error("HTTP %s from %s. Not retrying.", r.status_code, url)
            try:
                return r.json()
            except Exception:
                return {"text": r.text, "status_code": r.status_code}
    logging.error("Max retries exceeded for %s", url)
    return None


# -----------------------
# 1) Load .plt
# -----------------------
def load_plt(input_dir: str) -> pd.DataFrame:
    """
    Parse Geolife .plt files in input_dir.
    Returns DataFrame with columns: user_id, lat, lon, alt, timestamp (UTC)
    Assumptions: .plt lines are either:
       lat,lon,unused,alt,date,time
    or lat,lon,alt,timestamp_iso  (less common)
    """
    df = pd.read_csv("Data/big_data_latest.csv")
    df = df[["user_id", "latitude", "longitude", "altitude_feet", "datetime"]]
    df.columns = ["user_id", "lat", "lon", "alt", "timestamp"]

    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


# -----------------------
# 2) Detect stay points
# -----------------------
def haversine_m(lat1, lon1, lat2, lon2) -> float:
    """Haversine distance in meters (digit-by-digit reliable)."""
    # convert to radians
    rlat1 = math.radians(lat1)
    rlon1 = math.radians(lon1)
    rlat2 = math.radians(lat2)
    rlon2 = math.radians(lon2)
    dlat = rlat2 - rlat1
    dlon = rlon2 - rlon1
    a = math.sin(dlat / 2.0) ** 2 + math.cos(rlat1) * math.cos(rlat2) * (math.sin(dlon / 2.0) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    R = 6371000.0
    return R * c


def detect_stays(df_points: pd.DataFrame,
                 dwell_seconds: int = DWELL_SECONDS,
                 dist_max_m: float = DIST_MAX_METERS) -> pd.DataFrame:
    """
    Classic stay-point detection:
    - iterate points per user
    - for point i, find first j>i such that distance(i,j) > dist_max
      if time between i and j-1 >= dwell -> create stay from i..j-1
    Returns DataFrame: user_id, arrive_time, leave_time, lat, lon, point_count
    """
    stays = []
    for user, g in tqdm(df_points.groupby("user_id")):
        pts = g.reset_index(drop=True)
        n = len(pts)
        i = 0
        while i < n:
            j = i + 1
            while j < n:
                d = haversine_m(pts.loc[i, "lat"], pts.loc[i, "lon"], pts.loc[j, "lat"], pts.loc[j, "lon"])
                if d > dist_max_m:
                    # point j is outside radius relative to i
                    t_i = pts.loc[i, "timestamp"]
                    t_jm1 = pts.loc[j - 1, "timestamp"]
                    delta = (t_jm1 - t_i).total_seconds()
                    if delta >= dwell_seconds:
                        seg = pts.loc[i:(j - 1)]
                        mean_lat = seg["lat"].mean()
                        mean_lon = seg["lon"].mean()
                        arrives = t_i.isoformat()
                        leaves = t_jm1.isoformat()
                        stays.append({
                            "user_id": user,
                            "arrive_time": arrives,
                            "leave_time": leaves,
                            "lat": mean_lat,
                            "lon": mean_lon,
                            "point_count": len(seg)
                        })
                        i = j  # continue after j
                    else:
                        i += 1
                    break
                else:
                    j += 1
            else:
                # reached end
                t_i = pts.loc[i, "timestamp"]
                t_last = pts.loc[n - 1, "timestamp"]
                delta = (t_last - t_i).total_seconds()
                if delta >= dwell_seconds:
                    seg = pts.loc[i:(n - 1)]
                    stays.append({
                        "user_id": user,
                        "arrive_time": t_i.isoformat(),
                        "leave_time": t_last.isoformat(),
                        "lat": seg["lat"].mean(),
                        "lon": seg["lon"].mean(),
                        "point_count": len(seg)
                    })
                break
        # end while per user
    stays_df = pd.DataFrame(stays)
    if stays_df.empty:
        logging.warning("No stays detected with the provided thresholds.")
    return stays_df


# -----------------------
# 3) Query OSM / Overpass
# -----------------------
OVERPASS_POI_TAGS = [
    # common keys to search: amenity, shop, tourism, leisure, historic, office, healthcare, sport
    "amenity", "shop", "tourism", "leisure", "historic", "office", "healthcare", "sport", "building"
]


def build_overpass_query(lat: float, lon: float, radius_m: int, output_limit: int = 50) -> str:
    """
    Build an Overpass QL query to fetch nodes/ways/relations with useful tags within radius.
    """
    # We search for objects with any of the keys in OVERPASS_POI_TAGS
    key_filters = []
    for k in OVERPASS_POI_TAGS:
        key_filters.append(f'node(around:{radius_m},{lat},{lon})[{k}];')
        key_filters.append(f'way(around:{radius_m},{lat},{lon})[{k}];')
        key_filters.append(f'relation(around:{radius_m},{lat},{lon})[{k}];')
    body = (
        "[out:json][timeout:25];\n"
        "(" + "\n".join(key_filters) + ");\n"
        "out center %d;" % output_limit
    )
    return body


def query_overpass_pois(lat: float, lon: float, radius_m: int = QUERY_RADIUS_M) -> List[dict]:
    """
    Query Overpass API and return list of elements with tags and center coordinates.
    Uses safe_request wrapper (cache + throttle + backoff).
    """
    q = build_overpass_query(lat, lon, radius_m)
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json", "From": CONTACT_EMAIL}
    res = safe_request("POST", OVERPASS_ENDPOINT, data=q, headers=headers)
    if not res:
        return []
    elems = res.get("elements", [])
    results = []
    for el in elems:
        tags = el.get("tags", {}) or {}
        # get lat/lon: nodes have lat/lon, ways/relation have center
        if el.get("type") == "node":
            lat_e = el.get("lat")
            lon_e = el.get("lon")
        else:
            center = el.get("center") or {}
            lat_e = center.get("lat")
            lon_e = center.get("lon")
        if lat_e is None or lon_e is None:
            continue
        results.append({
            "id": el.get("id"),
            "type": el.get("type"),
            "lat": lat_e,
            "lon": lon_e,
            "tags": tags
        })
    return results


def query_nominatim_reverse(lat: float, lon: float) -> Optional[dict]:
    """
    Fallback: Nominatim reverse lookup for place name/address.
    Returns JSON with display_name and lat/lon.
    """
    params = {"lat": lat, "lon": lon, "format": "json", "addressdetails": 1}
    headers = {"User-Agent": USER_AGENT, "From": CONTACT_EMAIL}
    res = safe_request("GET", NOMINATIM_REVERSE, params=params, headers=headers)
    return res


# -----------------------
# 4) Tag -> category mapping
# -----------------------
# Tag to category mapping priorities
TAG_CATEGORY_MAP = {
    "restaurant": ["amenity=restaurant"],
    "cafe": ["amenity=cafe"],
    "school": ["amenity=school", "amenity=college", "amenity=university"],
    "park": ["leisure=park", "landuse=recreation_ground"],
    "hospital": ["amenity=hospital", "amenity=clinic", "healthcare=hospital"],
    "retail": ["shop", "amenity=shop"],
    "office": ["office", "building=office"],
    "residential": ["building=residential", "residential"],
    "transport": ["amenity=bus_station", "public_transport", "railway=station"],
    "worship": ["amenity=place_of_worship", "religion"],
    "lodging": ["tourism=hotel", "tourism=guest_house", "tourism=hostel"],
    "entertainment": ["amenity=theatre", "amenity=cinema", "tourism=attraction", "historic"],
    "government": ["government", "amenity=townhall"],
    "sports": ["sport", "leisure=sports_centre", "amenity=gym"],
    "other": []
}

# tag score lookup for confidence (simple)
TAG_SCORE_EXACT = 1.0
TAG_SCORE_PARTIAL = 0.7
TAG_SCORE_GENERIC = 0.4


def map_tags_to_category(tags: Dict[str, str]) -> Tuple[str, float, List[str]]:
    """
    Given OSM tags, return (category, tag_score, matched_keys)
    Simple rule: if any key=value exactly in mapping -> exact; else if key in mapping -> partial.
    """
    if not tags:
        return "Other", TAG_SCORE_GENERIC, []
    matched = []
    best_category = "Other"
    best_score = TAG_SCORE_GENERIC
    for cat, examples in TAG_CATEGORY_MAP.items():
        for ex in examples:
            if "=" in ex:
                k, v = ex.split("=", 1)
                if tags.get(k) == v:
                    matched.append(f"{k}={v}")
                    if TAG_SCORE_EXACT > best_score:
                        best_category = cat.title()
                        best_score = TAG_SCORE_EXACT
            else:
                # just key presence (partial)
                if ex in tags:
                    matched.append(ex)
                    if TAG_SCORE_PARTIAL > best_score:
                        best_category = cat.title()
                        best_score = TAG_SCORE_PARTIAL
    # fallback: if tag 'amenity' present but not mapped, mark as Other with partial score
    if not matched:
        if "amenity" in tags or "shop" in tags or "tourism" in tags:
            return "Other", TAG_SCORE_PARTIAL, []
    return best_category.title(), best_score, matched


# -----------------------
# 5) Matching & scoring
# -----------------------
def score_poi(stay_lat: float, stay_lon: float, poi_lat: float, poi_lon: float,
              query_radius_m: float, tag_score: float) -> Tuple[float, float]:
    """
    Compute match_distance_m and confidence_score.
    confidence = max(0, 1 - dist/radius) * tag_score
    """
    dist = haversine_m(stay_lat, stay_lon, poi_lat, poi_lon)
    conf = max(0.0, 1.0 - (dist / float(query_radius_m))) * tag_score
    return dist, conf


# -----------------------
# 6) Deduplicate POIs
# -----------------------
def dedupe_pois(pois: List[dict], dedupe_radius_m: float = POI_DEDUPE_RADIUS_M) -> List[dict]:
    """
    Simple greedy dedupe: sort by confidence desc, then keep if not within dedupe_radius of kept ones.
    Each poi must have 'lat','lon','confidence'
    """
    kept = []
    pois_sorted = sorted(pois, key=lambda x: x.get("confidence", 0.0), reverse=True)
    for p in pois_sorted:
        too_close = False
        for k in kept:
            d = haversine_m(p["lat"], p["lon"], k["lat"], k["lon"])
            if d <= dedupe_radius_m:
                too_close = True
                break
        if not too_close:
            kept.append(p)
    return kept


# -----------------------
# 7) Enrichment pipeline
# -----------------------
def enrich_stays(stays_df: pd.DataFrame,
                 query_radius_m: int = QUERY_RADIUS_M,
                 use_overpass: bool = True,
                 dedupe_radius_m: int = POI_DEDUPE_RADIUS_M) -> pd.DataFrame:
    """
    For each stay, query POIs and return enriched DataFrame matching OUTPUT_COLUMNS.
    """
    out_rows = []
    for idx, stay in tqdm(stays_df.iterrows()):
        s_lat = float(stay["lat"])
        s_lon = float(stay["lon"])
        user_id = stay["user_id"]
        arrive = stay["arrive_time"]
        leave = stay["leave_time"]

        pois_found = []
        # 1) try Overpass
        if use_overpass:
            try:
                elements = query_overpass_pois(s_lat, s_lon, radius_m=query_radius_m)
            except Exception as e:
                logging.warning("Overpass failed: %s", e)
                elements = []
            for e in elements:
                tags = e.get("tags", {})
                name = tags.get("name") or tags.get("official_name") or None
                poi_lat = e.get("lat")
                poi_lon = e.get("lon")
                cat, tag_score, matched_keys = map_tags_to_category(tags)
                dist, conf = score_poi(s_lat, s_lon, poi_lat, poi_lon, query_radius_m, tag_score)
                pois_found.append({
                    "name": name or tags.get("amenity") or tags.get("shop") or "Unknown",
                    "category": cat,
                    "lat": poi_lat,
                    "lon": poi_lon,
                    "tags": tags,
                    "matched_keys": matched_keys,
                    "distance": dist,
                    "confidence": conf,
                    "source": "overpass"
                })
        # 2) fallback: Nominatim reverse (if no POIs or as complementary)
        if not pois_found:
            nom = query_nominatim_reverse(s_lat, s_lon)
            if nom:
                name = nom.get("display_name")
                # approximate lat/lon in response
                poi_lat = float(nom.get("lat")) if nom.get("lat") else s_lat
                poi_lon = float(nom.get("lon")) if nom.get("lon") else s_lon
                tags = nom.get("address") or {}
                cat, tag_score, matched_keys = map_tags_to_category(tags)
                dist, conf = score_poi(s_lat, s_lon, poi_lat, poi_lon, query_radius_m, tag_score)
                pois_found.append({
                    "name": name,
                    "category": cat,
                    "lat": poi_lat,
                    "lon": poi_lon,
                    "tags": tags,
                    "matched_keys": matched_keys,
                    "distance": dist,
                    "confidence": conf,
                    "source": "nominatim"
                })

        # If still no POIs, produce a row with empty POI fields
        if not pois_found:
            out_rows.append({
                "user_id": user_id,
                "arrive_time": arrive,
                "leave_time": leave,
                "lat": s_lat,
                "lon": s_lon,
                "poi_name": "",
                "poi_category": "",
                "poi_lat": "",
                "poi_lon": "",
                "match_distance_m": "",
                "query_radius_m": query_radius_m,
                "confidence_score": 0.0,
                "source": ""
            })
            continue

        # Deduplicate found pois
        deduped = dedupe_pois(pois_found, dedupe_radius_m)

        # For each deduped poi, produce a row (could also choose only best match; here we output all)
        for p in deduped:
            out_rows.append({
                "user_id": user_id,
                "arrive_time": arrive,
                "leave_time": leave,
                "lat": s_lat,
                "lon": s_lon,
                "poi_name": p["name"],
                "poi_category": p["category"],
                "poi_lat": p["lat"],
                "poi_lon": p["lon"],
                "match_distance_m": round(p["distance"], 1),
                "query_radius_m": query_radius_m,
                "confidence_score": round(float(p["confidence"]), 3),
                "source": p["source"]
            })
    out_df = pd.DataFrame(out_rows)
    # ensure output columns order
    if not out_df.empty:
        out_df = out_df[OUTPUT_COLUMNS]
    return out_df


# -----------------------
# CLI / main
# -----------------------
def main():
    parser = argparse.ArgumentParser(description="Geolife stay detection + OSM POI enrichment")
    parser.add_argument("--input_dir", required=True, help="Directory with .plt files (recursively searched)")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--dwell_minutes", type=float, default=DWELL_SECONDS / 60.0, help="Dwell time in minutes")
    parser.add_argument("--dist_m", type=float, default=DIST_MAX_METERS, help="Max stay radius in meters")
    parser.add_argument("--query_radius_m", type=int, default=QUERY_RADIUS_M, help="POI query radius in meters")
    parser.add_argument("--use_overpass", action="store_true", default=True, help="Use Overpass API (default)")
    parser.add_argument("--no_cache", action="store_true", help="Disable cache (not recommended)")
    args = parser.parse_args()

    global MIN_REQUEST_INTERVAL
    if args.no_cache:
        # If disabling cache, set unique cache dir to temp to avoid reuse
        temp_dir = tempfile.mkdtemp(prefix="geolife_cache_")
        global CACHE_DIR
        CACHE_DIR = temp_dir

    logging.info("Loading .plt files from %s", args.input_dir)
    df_points = load_plt(args.input_dir)

    logging.info("Detecting stays (dwell %.1f min, radius %.1f m)", args.dwell_minutes, args.dist_m)
    stays = detect_stays(df_points, dwell_seconds=int(args.dwell_minutes * 60), dist_max_m=args.dist_m)
    if stays.empty:
        logging.warning("No stays found. Exiting.")
        stays.to_csv(args.output, index=False)
        return

    logging.info("Found %d stays. Enriching with POIs (radius %dm). This may take time.", len(stays), args.query_radius_m)
    enriched = enrich_stays(stays, query_radius_m=args.query_radius_m, use_overpass=args.use_overpass)
    logging.info("Writing output to %s", args.output)
    enriched.to_csv(args.output, index=False)
    logging.info("Done. Rows: %d", len(enriched))


if __name__ == "__main__":
    main()
