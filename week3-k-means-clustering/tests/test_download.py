from __future__ import annotations

import hashlib
import json
from pathlib import Path

from optical_fibre_clustering import download


def test_sha256_file(tmp_path: Path) -> None:
    sample = tmp_path / "sample.csv"
    sample.write_bytes(b"column\nvalue\n")

    assert download.sha256_file(sample) == hashlib.sha256(sample.read_bytes()).hexdigest()


def test_download_writes_data_and_metadata(monkeypatch, tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []
    file_contents = b"SNR,P1\n10,0.5\n"
    monkeypatch.setattr(download, "EXPECTED_SIZE_BYTES", len(file_contents))
    monkeypatch.setattr(download, "EXPECTED_SHA256", hashlib.sha256(file_contents).hexdigest())

    def fake_dataset_download(
        handle: str,
        *,
        path: str,
        output_dir: str,
        force_download: bool,
    ) -> str:
        calls.append(
            {
                "handle": handle,
                "path": path,
                "output_dir": output_dir,
                "force_download": force_download,
            }
        )
        destination = Path(output_dir) / path
        destination.write_bytes(file_contents)
        return str(destination)

    monkeypatch.setattr(download.kagglehub, "dataset_download", fake_dataset_download)

    data_path, metadata_path = download.download_dataset(tmp_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert data_path == tmp_path / download.DATASET_FILENAME
    assert calls[0]["handle"] == download.DATASET_HANDLE
    assert metadata["size_bytes"] == data_path.stat().st_size
    assert metadata["sha256"] == download.sha256_file(data_path)


def test_existing_data_is_reused_without_network(monkeypatch, tmp_path: Path) -> None:
    data_path = tmp_path / download.DATASET_FILENAME
    file_contents = b"SNR,P1\n10,0.5\n"
    data_path.write_bytes(file_contents)
    monkeypatch.setattr(download, "EXPECTED_SIZE_BYTES", len(file_contents))
    monkeypatch.setattr(download, "EXPECTED_SHA256", hashlib.sha256(file_contents).hexdigest())

    def fail_if_called(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("network downloader should not be called")

    monkeypatch.setattr(download.kagglehub, "dataset_download", fail_if_called)

    reused_path, metadata_path = download.download_dataset(tmp_path)

    assert reused_path == data_path
    assert metadata_path.exists()
