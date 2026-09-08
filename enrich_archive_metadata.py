import json
from pathlib import Path

import requests

ARCHIVE_JSON = Path("docs/archive.json")
USGS_EVENT_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"


def read_json(path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def fetch_event(event_id):
    response = requests.get(
        USGS_EVENT_URL,
        params={
            "format": "geojson",
            "eventid": event_id,
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def main():
    archive = read_json(ARCHIVE_JSON, [])
    changed = False

    for item in archive:
        if all(key in item for key in ("latitude", "longitude", "depth_km")):
            continue

        event_id = item.get("event_id")
        if not event_id:
            continue

        try:
            event = fetch_event(event_id)
            coords = (event.get("geometry") or {}).get("coordinates") or []

            if len(coords) < 3:
                print(f"Skipping {event_id}: no usable coordinates returned")
                continue

            lon, lat, depth_km = coords[:3]

            item["latitude"] = round(float(lat), 5)
            item["longitude"] = round(float(lon), 5)
            item["depth_km"] = round(float(depth_km or 0.0), 1)

            changed = True
            print(f"Enriched {event_id}")

        except Exception as exc:
            print(f"Unable to enrich {event_id}: {type(exc).__name__}: {exc}")

    if changed:
        archive.sort(key=lambda item: item.get("origin_utc", ""), reverse=True)
        write_json(ARCHIVE_JSON, archive)
        print("Archive metadata updated.")
    else:
        print("No archive metadata changes needed.")


if __name__ == "__main__":
    main()
