import json
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import requests
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.geodetics import gps2dist_azimuth, kilometer2degrees
from obspy.taup import TauPyModel

STATION = "R1C99"
NETWORK = "AM"
LOCATION = "00"
CHANNEL = "EHZ"

# Exact station coordinates are supplied only through encrypted
# GitHub Actions secrets. They are never written to public output.
STATION_LAT = float(os.environ["BRAMBLEY_LAT"])
STATION_LON = float(os.environ["BRAMBLEY_LON"])

PACIFIC = ZoneInfo("America/Los_Angeles")

OUTDIR = Path("docs")
ARCHIVE_DIR = OUTDIR / "archive"
ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
ARCHIVE_JSON = OUTDIR / "archive.json"
PROCESSED_JSON = OUTDIR / "archive-processed.json"

USGS_QUERY = "https://earthquake.usgs.gov/fdsnws/event/1/query"

# Broad Pacific Northwest region only. These are not Brambley's coordinates.
REGIONAL_MIN_LAT = 44.5
REGIONAL_MAX_LAT = 50.0
REGIONAL_MIN_LON = -126.5
REGIONAL_MAX_LON = -120.0

REGIONAL_MIN_MAG = 4.0
GLOBAL_MIN_MAG = 7.0
BACKFILL_YEARS = 6
MAX_EVENTS_PER_RUN = 8

REGIONAL_SNR_THRESHOLD = 4.0
GLOBAL_SNR_THRESHOLD = 3.0

TAUP = TauPyModel(model="iasp91")


def read_json(path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def usgs_query(params):
    r = requests.get(USGS_QUERY, params=params, timeout=60)
    r.raise_for_status()
    return r.json().get("features", [])


def fetch_candidates():
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=int(BACKFILL_YEARS * 365.25))

    common = {
        "format": "geojson",
        "starttime": start.isoformat().replace("+00:00", "Z"),
        "endtime": now.isoformat().replace("+00:00", "Z"),
        "orderby": "time",
        "limit": 20000,
    }

    regional_params = dict(common)
    regional_params.update({
        "minmagnitude": REGIONAL_MIN_MAG,
        "minlatitude": REGIONAL_MIN_LAT,
        "maxlatitude": REGIONAL_MAX_LAT,
        "minlongitude": REGIONAL_MIN_LON,
        "maxlongitude": REGIONAL_MAX_LON,
    })

    global_params = dict(common)
    global_params.update({"minmagnitude": GLOBAL_MIN_MAG})

    by_id = {}

    for feature in usgs_query(regional_params):
        props = feature.get("properties", {})
        if (props.get("type") or "earthquake").lower() != "earthquake":
            continue
        feature["_brambley_mode"] = "regional"
        by_id[feature["id"]] = feature

    for feature in usgs_query(global_params):
        props = feature.get("properties", {})
        if (props.get("type") or "earthquake").lower() != "earthquake":
            continue
        if feature["id"] not in by_id:
            feature["_brambley_mode"] = "global"
            by_id[feature["id"]] = feature

    candidates = list(by_id.values())
    candidates.sort(key=lambda f: f.get("properties", {}).get("time", 0), reverse=True)
    return candidates


def predicted_arrivals(event):
    lon, lat, depth_km = event["geometry"]["coordinates"][:3]
    depth_km = max(float(depth_km or 0.0), 0.0)

    distance_m, _, _ = gps2dist_azimuth(
        float(lat), float(lon), STATION_LAT, STATION_LON
    )
    distance_km = distance_m / 1000.0
    distance_deg = kilometer2degrees(distance_km)

    arrivals = TAUP.get_travel_times(
        source_depth_in_km=depth_km,
        distance_in_degree=distance_deg,
        phase_list=["P", "p", "Pn", "Pg", "S", "s", "Sn", "Sg"],
    )

    p_times = [a.time for a in arrivals if a.name.upper().startswith("P")]
    s_times = [a.time for a in arrivals if a.name.upper().startswith("S")]

    # TauP can return no local crustal phase at very small distances.
    # These conservative velocity fallbacks are only used for window placement.
    p_sec = min(p_times) if p_times else distance_km / 6.0
    s_sec = min(s_times) if s_times else distance_km / 3.5

    return float(p_sec), float(s_sec)


def get_window(event, mode, p_sec, s_sec):
    origin = UTCDateTime(event["properties"]["time"] / 1000.0)
    p_time = origin + p_sec
    s_time = origin + s_sec

    if mode == "regional":
        fetch_start = p_time - 90
        fetch_end = max(s_time + 120, p_time + 240)
        baseline_start = p_time - 60
        baseline_end = p_time - 10
        signal_start = p_time - 5
        signal_end = max(s_time + 90, p_time + 180)
    else:
        # For M7+ teleseisms, preserve a long window so later body and
        # surface-wave energy can be detected.
        fetch_start = p_time - 660
        fetch_end = p_time + 2700
        baseline_start = p_time - 600
        baseline_end = p_time - 120
        signal_start = p_time - 30
        signal_end = p_time + 2400

    return {
        "origin": origin,
        "fetch_start": fetch_start,
        "fetch_end": fetch_end,
        "baseline_start": baseline_start,
        "baseline_end": baseline_end,
        "signal_start": signal_start,
        "signal_end": signal_end,
    }


def fetch_waveform(window, mode):
    client = Client("RASPISHAKE")
    stream = client.get_waveforms(
        NETWORK,
        STATION,
        LOCATION,
        CHANNEL,
        window["fetch_start"],
        window["fetch_end"],
        attach_response=False,
    )

    if not stream:
        raise RuntimeError("No waveform returned")

    stream.merge(method=1, fill_value="interpolate")
    trace = stream[0]
    trace.detrend("linear")
    trace.detrend("demean")

    if mode == "regional":
        trace.filter("bandpass", freqmin=0.7, freqmax=15.0, corners=4, zerophase=True)
    else:
        trace.filter("bandpass", freqmin=0.03, freqmax=2.0, corners=4, zerophase=True)

    return trace


def slice_data(trace, start, end):
    tr = trace.copy()
    tr.trim(starttime=start, endtime=end, pad=False)
    if tr.stats.npts < 10:
        return np.array([], dtype=float)
    return np.asarray(tr.data, dtype=float)


def rms(values):
    if values.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(values))))


def robust_peak_rms(values, sampling_rate, seconds=8.0):
    if values.size == 0:
        return 0.0
    window = max(int(sampling_rate * seconds), 1)
    if values.size <= window:
        return rms(values)
    squared = np.square(values)
    kernel = np.ones(window, dtype=float) / window
    rolling = np.sqrt(np.convolve(squared, kernel, mode="valid"))
    return float(np.max(rolling))


def signal_to_noise(trace, window):
    baseline = slice_data(trace, window["baseline_start"], window["baseline_end"])
    signal = slice_data(trace, window["signal_start"], window["signal_end"])
    noise_rms = rms(baseline)
    signal_peak = robust_peak_rms(signal, trace.stats.sampling_rate, seconds=8.0)
    if noise_rms <= 0:
        return 0.0
    return signal_peak / noise_rms


def to_pacific(t):
    return t.datetime.replace(tzinfo=timezone.utc).astimezone(PACIFIC)


def decimate_for_plot(times, data, max_points=70000):
    if len(data) <= max_points:
        return times, data
    step = math.ceil(len(data) / max_points)
    return times[::step], data[::step]


def render_waveform(trace, event, mode, window, output_path):
    props = event["properties"]
    mag = float(props["mag"])
    place = props.get("place") or "Earthquake"

    trace_start = to_pacific(trace.stats.starttime)
    sample_times = np.array([
        trace_start + timedelta(seconds=float(s)) for s in trace.times()
    ], dtype=object)
    sample_times, data = decimate_for_plot(sample_times, np.asarray(trace.data))
    origin_dt = to_pacific(window["origin"])

    fig, ax = plt.subplots(figsize=(12, 3.7), dpi=160)
    ax.plot(sample_times, data, linewidth=0.65)

    # Predicted P/S times are used privately for detection only.
    # They are deliberately not published on the plot.
    ax.axvline(origin_dt, linewidth=1.0, alpha=0.45)
    ax.annotate(
        "Earthquake occurs",
        xy=(origin_dt, 0.96),
        xycoords=("data", "axes fraction"),
        xytext=(5, 0),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=8,
        alpha=0.70,
    )

    ax.set_title(f"M{mag:.1f} · {place}", fontsize=12)
    ax.set_xlabel("Pacific Time")
    ax.set_ylabel("Relative ground motion")
    ax.set_yticks([])
    ax.grid(True, linewidth=0.35, alpha=0.22)

    if mode == "regional":
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=1))
    else:
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=10))

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%-I:%M %p", tz=PACIFIC))
    fig.autofmt_xdate(rotation=0, ha="center")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def make_public_entry(event, mode, snr_value, filename):
    props = event["properties"]
    origin_utc = datetime.fromtimestamp(props["time"] / 1000.0, tz=timezone.utc)

    # No exact station coordinates, distance, azimuth, or predicted P/S
    # times are written to this public record.
    return {
        "event_id": event["id"],
        "magnitude": round(float(props["mag"]), 1),
        "place": props.get("place") or "Earthquake",
        "origin_utc": origin_utc.isoformat().replace("+00:00", "Z"),
        "usgs_url": props.get("url", ""),
        "scope": mode,
        "station": STATION,
        "channel": CHANNEL,
        "waveform": f"archive/{filename}",
        "signal_score": round(float(snr_value), 1),
    }


def process_candidate(event):
    mode = event["_brambley_mode"]
    props = event["properties"]
    event_time = datetime.fromtimestamp(props["time"] / 1000.0, tz=timezone.utc)

    if datetime.now(timezone.utc) - event_time < timedelta(minutes=45):
        return "deferred", None

    print(f'Checking {event["id"]}: M{props.get("mag")} {props.get("place")} [{mode}]')

    p_sec, s_sec = predicted_arrivals(event)
    window = get_window(event, mode, p_sec, s_sec)
    trace = fetch_waveform(window, mode)
    snr_value = signal_to_noise(trace, window)

    threshold = REGIONAL_SNR_THRESHOLD if mode == "regional" else GLOBAL_SNR_THRESHOLD
    print(f"  signal score {snr_value:.2f} (threshold {threshold:.2f})")

    if snr_value < threshold:
        return "rejected", {
            "reason": "signal_below_threshold",
            "signal_score": round(snr_value, 2),
        }

    filename = f'{event["id"]}.png'
    render_waveform(trace, event, mode, window, ARCHIVE_DIR / filename)
    return "accepted", make_public_entry(event, mode, snr_value, filename)


def main():
    archive = read_json(ARCHIVE_JSON, [])
    processed = read_json(PROCESSED_JSON, {})
    archived_ids = {item["event_id"] for item in archive}

    candidates = fetch_candidates()
    handled = 0
    changed = False

    for event in candidates:
        if handled >= MAX_EVENTS_PER_RUN:
            break

        event_id = event["id"]
        if event_id in archived_ids:
            continue

        existing = processed.get(event_id, {})
        if existing.get("status") in {"accepted", "rejected"}:
            continue

        try:
            status, payload = process_candidate(event)
            if status == "deferred":
                continue

            handled += 1

            if status == "accepted":
                archive.append(payload)
                archived_ids.add(event_id)
                processed[event_id] = {
                    "status": "accepted",
                    "processed_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                    "signal_score": payload["signal_score"],
                }
                changed = True

            elif status == "rejected":
                processed[event_id] = {
                    "status": "rejected",
                    "processed_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                    **(payload or {}),
                }
                changed = True

        except Exception as exc:
            handled += 1
            print(f"  ERROR: {type(exc).__name__}: {exc}")
            # Errors remain retryable on later runs.
            processed[event_id] = {
                "status": "error",
                "processed_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "error": f"{type(exc).__name__}: {exc}",
            }
            changed = True

    archive.sort(key=lambda item: item.get("origin_utc", ""), reverse=True)

    if changed or not ARCHIVE_JSON.exists():
        write_json(ARCHIVE_JSON, archive)
        write_json(PROCESSED_JSON, processed)

    print(f"Archive now contains {len(archive)} event(s).")


if __name__ == "__main__":
    main()
