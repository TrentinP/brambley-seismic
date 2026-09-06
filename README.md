# Brambley Observatory — Stage Three

This starter project automatically:
1. Queries USGS for the newest M3.0+ earthquake inside a broad Pacific Northwest box.
2. Waits until the event is old enough for the Raspberry Shake historical archive.
3. Downloads R1C99 AM.00.EHZ waveform data through the Raspberry Shake FDSN service.
4. Filters and renders a six-minute seismogram.
5. Publishes `docs/latest-waveform.png` and `docs/latest.json`.
6. Runs automatically twice each hour with GitHub Actions.

No Brambley coordinates are used by the script.

## GitHub setup

Create a new public repository named `brambley-seismic`.

Upload the contents of this folder, preserving `.github/workflows/update-waveform.yml`.

In the repository, open Settings > Actions > General and ensure workflows have Read and write permissions.

Run the workflow once from Actions > Update Brambley earthquake waveform > Run workflow.

After it succeeds, enable GitHub Pages:
Settings > Pages > Deploy from a branch > main > /docs.

Your public files will then normally be available at:
https://YOUR-GITHUB-NAME.github.io/brambley-seismic/latest-waveform.png
https://YOUR-GITHUB-NAME.github.io/brambley-seismic/latest.json

Paste `squarespace-code.html` into a Squarespace Code Block and replace YOURNAME with your GitHub username.
