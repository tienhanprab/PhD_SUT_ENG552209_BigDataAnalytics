"""Fetch the pinned public Kaggle CSV and verify its exact content."""
import hashlib
import io
import json
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
URL = 'https://www.kaggle.com/api/v1/datasets/download/isurumy93/solar-pv-anomaly-detection-dataset?datasetVersionNumber=2'
SHA256 = '3a36265b808df72ca2feac66f417ffe01c574795437dd17dad5bd49ff63f08ba'


def download():
    destination = ROOT / 'data/raw/pv_dataset.csv'
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        content = destination.read_bytes()
    else:
        with urllib.request.urlopen(URL, timeout=90) as response:
            archive = zipfile.ZipFile(io.BytesIO(response.read()))
        content = archive.read('pv_dataset.csv')
    if hashlib.sha256(content).hexdigest() != SHA256:
        raise ValueError('CSV checksum differs from reviewed version 2; refusing to continue')
    if not destination.exists():
        destination.write_bytes(content)
    metadata = dict(dataset='isurumy93/solar-pv-anomaly-detection-dataset', version=2,
                    url=URL, filename=destination.name, sha256=SHA256, bytes=len(content),
                    verified_at_utc=datetime.now(timezone.utc).isoformat())
    (ROOT / 'data/raw/provenance.json').write_text(json.dumps(metadata, indent=2))
    return destination


if __name__ == '__main__':
    print(download())
