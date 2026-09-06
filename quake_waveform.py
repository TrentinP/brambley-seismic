import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import requests
from obspy import UTCDateTime
from obspy.clients.fdsn import Client


# ---------------------------------------------------------
# RASPBERRY SHAKE
# ---------------------------------------------------------

STATION = "R1C99"
NETWORK = "AM"
LOCATION = "00"
CHANNEL = "EHZ"


# ---------------------------------------------------------
# REGIONAL EARTHQUAKE SEARCH
#
# This is only a broad Pacific Northwest search area.
# These coordinates do NOT represent Brambley's location.
# ---------------------------------------------------------

MIN_LAT = 44.5
MAX_LAT = 50.0

MIN_LON = -126.5
MAX_LON = -120.0

MIN_MAGNITUDE = 3.0
LOOKBACK_DAYS = 7


# ---------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------

OUTDIR = Path("docs")
OUTDIR.mkdir(exist_ok=True)


# ---------------------------------------------------------
# USGS
# ---------------------------------------------------------

USGS_QUERY = (
    "https://earthquake.usgs.gov/"
    "fdsnws/event/1/query"
)


# ---------------------------------------------------------
# TIMEZONE
# ---------------------------------------------------------

PACIFIC = ZoneInfo("America/Los_Angeles")


# ---------------------------------------------------------
# FIND NEWEST QUALIFYING EARTHQUAKE
# ---------------------------------------------------------

def fetch_latest_qualifying_event():

    end = datetime.now(timezone.utc)

    start = (
        end
        - timedelta(days=LOOKBACK_DAYS)
    )

    params = {
        "format": "geojson",

        "starttime":
            start.isoformat()
            .replace("+00:00", "Z"),

        "endtime":
            end.isoformat()
            .replace("+00:00", "Z"),

        "minmagnitude":
            MIN_MAGNITUDE,

        "minlatitude":
            MIN_LAT,

        "maxlatitude":
            MAX_LAT,

        "minlongitude":
            MIN_LON,

        "maxlongitude":
            MAX_LON,

        "orderby":
            "time",

        "limit":
            20
    }

    response = requests.get(
        USGS_QUERY,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    features = (
        response.json()
        .get("features", [])
    )

    # Ignore quarry blasts and explosions
    # when USGS identifies them as such.

    for feature in features:

        props = feature.get(
            "properties",
            {}
        )

        event_type = (
            props.get("type")
            or "earthquake"
        ).lower()

        if event_type == "earthquake":
            return feature

    return None


# ---------------------------------------------------------
# GET R1C99 WAVEFORM
# ---------------------------------------------------------

def fetch_waveform(event_time_ms):

    origin = UTCDateTime(
        event_time_ms / 1000.0
    )

    # Public display window:
    #
    # 30 sec before earthquake origin
    # 150 sec after earthquake origin
    #
    # Total = 3 minutes

    start = origin - 30
    end = origin + 150

    client = Client(
        "RASPISHAKE"
    )

    stream = client.get_waveforms(
        NETWORK,
        STATION,
        LOCATION,
        CHANNEL,
        start,
        end,
        attach_response=False
    )

    stream.merge(
        method=1,
        fill_value="interpolate"
    )

    trace = stream[0]

    # Remove baseline and linear trend.

    trace.detrend("linear")
    trace.detrend("demean")

    # Useful display band for local/regional earthquakes.

    trace.filter(
        "bandpass",
        freqmin=0.7,
        freqmax=15.0,
        corners=4,
        zerophase=True
    )

    return (
        trace,
        origin,
        start,
        end
    )


# ---------------------------------------------------------
# MAKE PUBLIC SEISMOGRAM
# ---------------------------------------------------------

def make_plot(
    trace,
    event,
    origin,
    start,
    end
):

    props = event["properties"]

    magnitude = float(
        props["mag"]
    )

    place = (
        props.get("place")
        or "Regional earthquake"
    )

    event_id = event["id"]

    usgs_url = (
        props.get("url", "")
    )


    # -----------------------------------------------------
    # START AND END TIMES IN PACIFIC TIME
    # -----------------------------------------------------

    start_dt = (
        start.datetime
        .replace(
            tzinfo=timezone.utc
        )
        .astimezone(PACIFIC)
    )

    end_dt = (
        end.datetime
        .replace(
            tzinfo=timezone.utc
        )
        .astimezone(PACIFIC)
    )


    # -----------------------------------------------------
    # BUILD CLOCK TIMES FOR EACH SAMPLE
    # -----------------------------------------------------

    sample_seconds = trace.times()

    sample_times = [

        start_dt
        + timedelta(
            seconds=float(sec)
        )

        for sec in sample_seconds
    ]


    # -----------------------------------------------------
    # EARTHQUAKE ORIGIN TIME
    # -----------------------------------------------------

    origin_datetime = (

        origin.datetime
        .replace(
            tzinfo=timezone.utc
        )
        .astimezone(PACIFIC)
    )


    # -----------------------------------------------------
    # CREATE GRAPH
    # -----------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(12, 3.6),
        dpi=160
    )

    ax.plot(
        sample_times,
        trace.data,
        linewidth=0.7
    )


    # -----------------------------------------------------
    # EARTHQUAKE ORIGIN-TIME LINE
    # -----------------------------------------------------

    ax.axvline(
        origin_datetime,
        linewidth=1.0,
        alpha=0.45
    )


    # Public-facing label.

    ax.annotate(

        "Earthquake occurs",

        xy=(
            origin_datetime,
            0.96
        ),

        xycoords=(
            "data",
            "axes fraction"
        ),

        xytext=(5, 0),

        textcoords="offset points",

        ha="left",
        va="top",

        fontsize=8,
        alpha=0.70
    )


    # -----------------------------------------------------
    # X AXIS — PACIFIC CLOCK TIME
    #
    # Force ticks to begin at the actual start of the
    # recording rather than allowing Matplotlib to choose
    # its own 30-second alignment.
    # -----------------------------------------------------

    tick_times = []

    tick = start_dt

    while tick <= end_dt:

        tick_times.append(tick)

        tick += timedelta(
            seconds=30
        )

    ax.set_xticks(
        tick_times
    )

    ax.xaxis.set_major_formatter(

        mdates.DateFormatter(
            "%-I:%M:%S %p",
            tz=PACIFIC
        )
    )


    # -----------------------------------------------------
    # LABELS
    # -----------------------------------------------------

    ax.set_xlabel(
        "Pacific Time"
    )

    ax.set_ylabel(
        "Relative ground motion"
    )

    ax.set_title(
        f"M{magnitude:.1f} · {place}",
        fontsize=12
    )


    # -----------------------------------------------------
    # CLEAN UP Y AXIS
    #
    # This is intentionally a visual seismogram rather
    # than a calibrated engineering-motion product.
    # -----------------------------------------------------

    ax.set_yticks([])


    # -----------------------------------------------------
    # GRID
    # -----------------------------------------------------

    ax.grid(
        True,
        linewidth=0.35,
        alpha=0.22
    )


    # -----------------------------------------------------
    # LIMIT GRAPH EXACTLY TO REQUESTED WINDOW
    # -----------------------------------------------------

    ax.set_xlim(
        start_dt,
        end_dt
    )


    # Keep the clock labels horizontal.

    fig.autofmt_xdate(
        rotation=0,
        ha="center"
    )

    fig.tight_layout()


    # -----------------------------------------------------
    # SAVE GRAPH
    # -----------------------------------------------------

    fig.savefig(
        OUTDIR / "latest-waveform.png",
        bbox_inches="tight"
    )

    plt.close(fig)


    # -----------------------------------------------------
    # JSON METADATA FOR SQUARESPACE
    # -----------------------------------------------------

    utc_dt = datetime.fromtimestamp(
        props["time"] / 1000.0,
        tz=timezone.utc
    )

    metadata = {

        "event_id":
            event_id,

        "magnitude":
            round(
                magnitude,
                1
            ),

        "place":
            place,

        "origin_utc":
            utc_dt
            .isoformat()
            .replace(
                "+00:00",
                "Z"
            ),

        "usgs_url":
            usgs_url,

        "station":
            STATION,

        "channel":
            CHANNEL,

        "image":
            "latest-waveform.png",

        "updated_utc":
            datetime.now(
                timezone.utc
            )
            .isoformat()
            .replace(
                "+00:00",
                "Z"
            )
    }

    (
        OUTDIR / "latest.json"
    ).write_text(

        json.dumps(
            metadata,
            indent=2
        ),

        encoding="utf-8"
    )


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    event = (
        fetch_latest_qualifying_event()
    )

    if not event:

        print(
            "No qualifying regional "
            "earthquake found."
        )

        return

    props = event[
        "properties"
    ]

    event_time = (
        datetime.fromtimestamp(
            props["time"] / 1000.0,
            tz=timezone.utc
        )
    )


    # Raspberry Shake historical data generally lags
    # real time. Give the archive approximately
    # 40 minutes before requesting it.

    if (
        datetime.now(
            timezone.utc
        )
        - event_time
        < timedelta(
            minutes=40
        )
    ):

        print(
            "Latest qualifying earthquake "
            "is still too recent for "
            "reliable archive access."
        )

        return


    print(
        f'Processing {event["id"]}: '
        f'M{props["mag"]} '
        f'{props.get("place")}'
    )


    (
        trace,
        origin,
        start,
        end
    ) = fetch_waveform(
        props["time"]
    )


    make_plot(
        trace,
        event,
        origin,
        start,
        end
    )


# ---------------------------------------------------------
# RUN
# ---------------------------------------------------------

if __name__ == "__main__":
    main()
