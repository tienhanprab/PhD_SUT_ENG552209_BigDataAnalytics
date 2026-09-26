from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from solar_apriori import download


def _csv_bytes(columns: tuple[str, ...], value: str = "1") -> bytes:
    header = ",".join(columns)
    row = ",".join(value for _ in columns)
    return f"{header}\n{row}\n".encode()


def _small_manifest() -> tuple[dict[str, download.FileSpec], dict[str, bytes]]:
    contents: dict[str, bytes] = {}
    manifest: dict[str, download.FileSpec] = {}
    for filename, original in download.EXPECTED_FILES.items():
        payload = _csv_bytes(original.required_columns, value=str(original.plant))
        contents[filename] = payload
        manifest[filename] = download.FileSpec(
            size_bytes=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
            required_columns=original.required_columns,
            plant=original.plant,
            role=original.role,
        )
    return manifest, contents


def test_pinned_version_one_manifest() -> None:
    assert download.DATASET_HANDLE == (
        "anikannal/solar-power-generation-data/versions/1"
    )
    assert download.DATASET_FILENAMES == (
        "Plant_1_Generation_Data.csv",
        "Plant_1_Weather_Sensor_Data.csv",
        "Plant_2_Generation_Data.csv",
        "Plant_2_Weather_Sensor_Data.csv",
    )
    assert download.EXPECTED_FILES["Plant_1_Generation_Data.csv"].size_bytes == 4_839_076
    assert download.EXPECTED_FILES["Plant_2_Weather_Sensor_Data.csv"].sha256 == (
        "cafd80885a51b88521b3301cffa4e4f013e2516fcb3478b439a81855edac3a64"
    )


def test_sha256_file(tmp_path: Path) -> None:
    sample = tmp_path / "sample.csv"
    sample.write_bytes(b"column\nvalue\n")

    assert download.sha256_file(sample) == hashlib.sha256(sample.read_bytes()).hexdigest()


def test_validate_file_rejects_missing_required_column(tmp_path: Path) -> None:
    payload = b"DATE_TIME,PLANT_ID\n2020-01-01,1\n"
    sample = tmp_path / "sample.csv"
    sample.write_bytes(payload)
    spec = download.FileSpec(
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        required_columns=("DATE_TIME", "PLANT_ID", "SOURCE_KEY"),
        plant=1,
        role="generation",
    )

    with pytest.raises(ValueError, match="SOURCE_KEY"):
        download.validate_file(sample, spec)


def test_download_stages_files_and_writes_provenance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest, contents = _small_manifest()
    monkeypatch.setattr(download, "EXPECTED_FILES", manifest)
    calls: list[dict[str, object]] = []

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
        staged_path = Path(output_dir) / path
        staged_path.write_bytes(contents[path])
        return str(staged_path)

    paths, metadata_path = download.download_dataset(
        tmp_path / "raw", downloader=fake_dataset_download
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert [call["path"] for call in calls] == list(manifest)
    assert all(call["handle"] == download.DATASET_HANDLE for call in calls)
    assert all(Path(str(call["output_dir"])) != tmp_path / "raw" for call in calls)
    assert all(paths[name].read_bytes() == contents[name] for name in manifest)
    assert metadata["dataset_version"] == 1
    assert {record["filename"] for record in metadata["files"]} == set(manifest)
    assert {record["acquisition_this_run"] for record in metadata["files"]} == {
        "downloaded"
    }


def test_existing_valid_files_are_reused_without_downloader(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest, contents = _small_manifest()
    monkeypatch.setattr(download, "EXPECTED_FILES", manifest)
    output_dir = tmp_path / "raw"
    output_dir.mkdir()
    for filename, payload in contents.items():
        (output_dir / filename).write_bytes(payload)

    def fail_if_called(*args: object, **kwargs: object) -> str:
        raise AssertionError("network downloader should not be called")

    paths, metadata_path = download.download_dataset(
        output_dir, downloader=fail_if_called
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert set(paths) == set(manifest)
    assert all(record["acquisition_this_run"] == "reused" for record in metadata["files"])


def test_force_validation_failure_keeps_existing_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest, contents = _small_manifest()
    monkeypatch.setattr(download, "EXPECTED_FILES", manifest)
    output_dir = tmp_path / "raw"
    output_dir.mkdir()
    for filename, payload in contents.items():
        (output_dir / filename).write_bytes(payload)

    first_filename = next(iter(manifest))

    def corrupt_download(
        handle: str,
        *,
        path: str,
        output_dir: str,
        force_download: bool,
    ) -> str:
        del handle, force_download
        staged_path = Path(output_dir) / path
        staged_path.write_bytes(b"corrupt\n")
        return str(staged_path)

    with pytest.raises(ValueError, match="Unexpected size"):
        download.download_dataset(output_dir, force=True, downloader=corrupt_download)

    assert (output_dir / first_filename).read_bytes() == contents[first_filename]

