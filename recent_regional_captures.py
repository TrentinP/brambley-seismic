import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from archive_earthquakes import (
    CHANNEL,
    REGIONAL_MAX_LAT,
    REGIONAL_MAX_LON,
    REGIONAL_MIN_LAT,
    REGIONAL_MIN_LON,
    REGIONAL_SNR_THRESHOLD,
    STATION,
    fetch_waveform,
    get_window,
    predicted_arrivals,
    render_waveform,
    signal_to_noise,
    usgs_query,
)

OUTDIR = Path("docs")
RECENT_DIR = OUTDIR / "recent-captures"
RECENT_DIR.mkdir(parents=True, exist_ok=True)
RECENT_JSON = OUTDIR / "recent-captures.json"
PROCESSED_JSON = OUTDIR / "recent-captures-processed.json"

MIN_MAG = 3.0
LOOKBACK_DAYS = 8
PUBLIC_WINDOW_DAYS = 7
MAX_EVENTS_PER_RUN = 10
RETRY_ERROR_HOURS = 6
RETRY_REJECTED_HOURS = 18


def read_json(path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def utc_iso(dt):
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_utc(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


def fetch_recent_candidates():
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=LOOKBACK_DAYS)

    params = {
        "format": "geojson",
        "starttime": utc_iso(start),
        "endtime": utc_iso(now),
        "orderby": "time",
        "minmagnitude": MIN_MAG,
        "minlatitude": REGIONAL_MIN_LAT,
        "maxlatitude": REGIONAL_MAX_LAT,
        "minlongitude": REGIONAL_MIN_LON,
        "maxlongitude": REGIONAL_MAX_LON,
        "limit": 2000,
    }

    features = usgs_query(params)

    candidates = []
    for feature in features:
        props = feature.get("properties", {})
        if (props.get("type") or "earthquake").lower() != "earthquake":
            continue
        feature["_brambley_mode"] = "regional"
        candidates.append(feature)

    candidates.sort(
        key=lambda f: f.get("properties", {}).get("time", 0),
        reverse=True,
    )
    return candidates


def should_retry(record, now):
    if not record:
        return True

    status = record.get("status")
    processed_at = parse_utc(record.get("processed_utc"))

    if status == "accepted":
        return False

    if processed_at is None:
        return True

    age = now - processed_at

    if status == "error":
        return age >= timedelta(hours=RETRY_ERROR_HOURS)

    if status == "rejected":
        return age >= timedelta(hours=RETRY_REJECTED_HOURS)

    return True


def make_public_entry(event, snr_value, filename):
    props = event["properties"]
    origin_utc = datetime.fromtimestamp(
        props["time"] / 1000.0,
        tz=timezone.utc,
    )

    # This public record intentionally contains no station coordinates,
    # distance, bearing, azimuth, or predicted arrival times.
    return {
        "event_id": event["id"],
        "magnitude": round(float(props["mag"]), 1),
        "place": props.get("place") or "Earthquake",
        "origin_utc": utc_iso(origin_utc),
        "usgs_url": props.get("url", ""),
        "station": STATION,
        "channel": CHANNEL,
        "waveform": f"recent-captures/{filename}",
        "signal_score": round(float(snr_value), 1),
    }


def process_candidate(event):
    props = event["properties"]
    event_time = datetime.fromtimestamp(
        props["time"] / 1000.0,
        tz=timezone.utc,
    )

    if datetime.now(timezone.utc) - event_time < timedelta(minutes=45):
        return "deferred", None

    print(
        f'Checking recent regional event {event["id"]}: '
        f'M{props.get("mag")} {props.get("place")}'
    )

    p_sec, s_sec = predicted_arrivals(event)
    window = get_window(event, "regional", p_sec, s_sec)
    trace = fetch_waveform(window, "regional")
    snr_value = signal_to_noise(trace, window)

    print(
        f"  signal score {snr_value:.2f} "
        f"(threshold {REGIONAL_SNR_THRESHOLD:.2f})"
    )

    if snr_value < REGIONAL_SNR_THRESHOLD:
        return "rejected", {
            "reason": "signal_below_threshold",
            "signal_score": round(snr_value, 2),
        }

    filename = f'{event["id"]}.png'
    render_waveform(
        trace,
        event,
        "regional",
        window,
        RECENT_DIR / filename,
    )

    return "accepted", make_public_entry(event, snr_value, filename)


def cleanup(captures, processed, now):
    cutoff = now - timedelta(days=LOOKBACK_DAYS + 1)

    keep_captures = []
    keep_ids = set()

    for item in captures:
        origin = parse_utc(item.get("origin_utc"))
        if origin is not None and origin >= cutoff:
            keep_captures.append(item)
            keep_ids.add(item.get("event_id"))

    keep_processed = {}
    for event_id, record in processed.items():
        processed_at = parse_utc(record.get("processed_utc"))
        if processed_at is not None and processed_at >= cutoff:
            keep_processed[event_id] = record

    for image_path in RECENT_DIR.glob("*.png"):
        if image_path.stem not in keep_ids:
            try:
                image_path.unlink()
            except OSError:
                pass

    return keep_captures, keep_processed


def main():
    now = datetime.now(timezone.utc)

    captures = read_json(RECENT_JSON, [])
    processed = read_json(PROCESSED_JSON, {})

    captures, processed = cleanup(captures, processed, now)
    captured_by_id = {
        item["event_id"]: item
        for item in captures
        if item.get("event_id")
    }

    candidates = fetch_recent_candidates()
    handled = 0

    for event in candidates:
        if handled >= MAX_EVENTS_PER_RUN:
            break

        event_id = event["id"]

        if event_id in captured_by_id:
            continue

        if not should_retry(processed.get(event_id), now):
            continue

        try:
            status, payload = process_candidate(event)

            if status == "deferred":
                continue

            handled += 1

            if status == "accepted":
                captured_by_id[event_id] = payload
                processed[event_id] = {
                    "status": "accepted",
                    "processed_utc": utc_iso(datetime.now(timezone.utc)),
                    "signal_score": payload["signal_score"],
                }

            elif status == "rejected":
                processed[event_id] = {
                    "status": "rejected",
                    "processed_utc": utc_iso(datetime.now(timezone.utc)),
                    **(payload or {}),
                }

        except Exception as exc:
            handled += 1
            print(f"  ERROR: {type(exc).__name__}: {exc}")
            processed[event_id] = {
                "status": "error",
                "processed_utc": utc_iso(datetime.now(timezone.utc)),
                "error": f"{type(exc).__name__}: {exc}",
            }

    public_cutoff = now - timedelta(days=PUBLIC_WINDOW_DAYS)

    public_captures = []
    for item in captured_by_id.values():
        origin = parse_utc(item.get("origin_utc"))
        if origin is not None and origin >= public_cutoff:
            public_captures.append(item)

    public_captures.sort(
        key=lambda item: item.get("origin_utc", ""),
        reverse=True,
    )

    # The published JSON contains only successfully captured events from
    # the current seven-day window. No station-location-derived fields
    # are exposed.
    write_json(RECENT_JSON, public_captures)
    write_json(PROCESSED_JSON, processed)

    print(
        f"Recent capture feed contains {len(public_captures)} "
        "successfully recorded M3.0+ event(s)."
    )


if __name__ == "__main__":
    main()
