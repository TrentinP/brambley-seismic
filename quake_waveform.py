import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import requests
from obspy import UTCDateTime, read
from obspy.clients.fdsn import Client

# ---- Public configuration ----
STATION = "R1C99"
NETWORK = "AM"
LOCATION = "00"
CHANNEL = "EHZ"

# Broad Pacific Northwest box only.
# These coordinates do NOT represent Brambley's location.
MIN_LAT, MAX_LAT = 44.5, 50.0
MIN_LON, MAX_LON = -126.5, -120.0

MIN_MAGNITUDE = 3.0
LOOKBACK_DAYS = 7

OUTDIR = Path("docs")
OUTDIR.mkdir(exist_ok=True)

USGS_QUERY = "https://earthquake.usgs.gov/fdsnws/event/1/query"

def fetch_latest_qualifying_event():
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=LOOKBACK_DAYS)

    params = {
        "format": "geojson",
        "starttime": start.isoformat().replace("+00:00", "Z"),
        "endtime": end.isoformat().replace("+00:00", "Z"),
        "minmagnitude": MIN_MAGNITUDE,
        "minlatitude": MIN_LAT,
        "maxlatitude": MAX_LAT,
        "minlongitude": MIN_LON,
        "maxlongitude": MAX_LON,
        "orderby": "time",
        "limit": 20,
    }

    r = requests.get(USGS_QUERY, params=params, timeout=30)
    r.raise_for_status()
    features = r.json().get("features", [])

    # Exclude quarry blasts/explosions where USGS identifies them as such.
    for feature in features:
        props = feature.get("properties", {})
        event_type = (props.get("type") or "earthquake").lower()
        if event_type == "earthquake":
            return feature

    return None

def fetch_waveform(event_time_ms):
    origin = UTCDateTime(event_time_ms / 1000.0)

    # For regional events this broad window reliably includes arrivals without
    # requiring or publishing Brambley's precise coordinates.
    start = origin - 30
    end = origin + 330

    client = Client("RASPISHAKE")
    st = client.get_waveforms(
        NETWORK, STATION, LOCATION, CHANNEL,
        start, end,
        attach_response=False
    )

    st.merge(method=1, fill_value="interpolate")
    tr = st[0]
    tr.detrend("linear")
    tr.detrend("demean")

    # A useful display band for local/regional earthquakes on the RS4D geophone.
    tr.filter("bandpass", freqmin=0.7, freqmax=15.0, corners=4, zerophase=True)

    return tr, origin, start, end

def make_plot(tr, event, origin, start, end):
    props = event["properties"]
    mag = float(props["mag"])
    place = props.get("place") or "Regional earthquake"
    event_id = event["id"]
    usgs_url = props.get("url", "")

    times = tr.times(reftime=start)
    data = tr.data

    fig, ax = plt.subplots(figsize=(12, 3.4), dpi=160)
    ax.plot(times, data, linewidth=0.65)
    ax.axvline(float(origin - start), linewidth=1.0, alpha=0.45)

    ax.set_xlim(0, float(end - start))
    ax.set_xlabel("Seconds from start of recording")
    ax.set_ylabel("Relative ground motion")
    ax.set_title(f"M{mag:.1f} · {place}\nRecorded by Raspberry Shake {STATION} · {CHANNEL}")
    ax.grid(True, linewidth=0.35, alpha=0.25)

    # Raw counts are intentionally unlabeled numerically; this is a visual
    # seismogram rather than a calibrated engineering-motion product.
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.set_yticklabels([])

    fig.tight_layout()
    fig.savefig(OUTDIR / "latest-waveform.png", bbox_inches="tight")
    plt.close(fig)

    utc_dt = datetime.fromtimestamp(props["time"] / 1000.0, tz=timezone.utc)

    metadata = {
        "event_id": event_id,
        "magnitude": round(mag, 1),
        "place": place,
        "origin_utc": utc_dt.isoformat().replace("+00:00", "Z"),
        "usgs_url": usgs_url,
        "station": STATION,
        "channel": CHANNEL,
        "image": "latest-waveform.png",
        "updated_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }

    (OUTDIR / "latest.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8"
    )

def main():
    event = fetch_latest_qualifying_event()
    if not event:
        print("No qualifying regional earthquake found.")
        return

    props = event["properties"]
    event_time = datetime.fromtimestamp(props["time"] / 1000.0, tz=timezone.utc)

    # Raspberry Shake's FDSN archive is historical (roughly T-30 min and older).
    # If the event is too fresh, leave the existing plot in place and try later.
    if datetime.now(timezone.utc) - event_time < timedelta(minutes=40):
        print("Latest qualifying earthquake is still too recent for reliable archive access.")
        return

    print(f'Processing {event["id"]}: M{props["mag"]} {props.get("place")}')

    tr, origin, start, end = fetch_waveform(props["time"])
    make_plot(tr, event, origin, start, end)

if __name__ == "__main__":
    main()
