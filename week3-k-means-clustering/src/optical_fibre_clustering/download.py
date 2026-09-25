"""Download the Optical Fibre Fault Detection dataset from Kaggle."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import kagglehub

DATASET_SLUG = "yogi2727/optical-fibre-fault-detection"
DATASET_VERSION = 1
DATASET_HANDLE = f"{DATASET_SLUG}/versions/{DATASET_VERSION}"
DATASET_FILENAME = "OTDR_data.csv"
DATASET_URL = f"https://www.kaggle.com/datasets/{DATASET_SLUG}"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"
METADATA_FILENAME = "OTDR_data.metadata.json"
EXPECTED_SIZE_BYTES = 74_389_769
EXPECTED_SHA256 = "aa2fbf1fe327686483d6c1a99ceefcd87e11d152a9d869b899cfe2132900b71f"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 checksum of a file without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_source_file(data_path: Path) -> str:
    """Validate the pinned Kaggle v1 source file and return its checksum."""
    actual_size = data_path.stat().st_size
    if actual_size != EXPECTED_SIZE_BYTES:
        raise ValueError(
            f"Unexpected dataset size: {actual_size:,} bytes; "
            f"expected {EXPECTED_SIZE_BYTES:,} bytes for Kaggle version {DATASET_VERSION}."
        )

    actual_sha256 = sha256_file(data_path)
    if actual_sha256 != EXPECTED_SHA256:
        raise ValueError(
            f"Unexpected dataset checksum: {actual_sha256}; expected {EXPECTED_SHA256}."
        )
    return actual_sha256


def _display_path(data_path: Path) -> str:
    try:
        return str(data_path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(data_path)


def _write_metadata(
    data_path: Path, metadata_path: Path, checksum: str
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "dataset_handle": DATASET_HANDLE,
        "dataset_version": DATASET_VERSION,
        "dataset_url": DATASET_URL,
        "source_file": DATASET_FILENAME,
        "local_file": _display_path(data_path),
        "downloaded_at_utc": datetime.now(UTC).isoformat(),
        "size_bytes": data_path.stat().st_size,
        "sha256": checksum,
        "license_as_reported_by_kaggle": "Unknown",
        "downloader": f"kagglehub {version('kagglehub')}",
    }

    temporary_path = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(metadata_path)
    return metadata


def download_dataset(
    output_dir: Path = DEFAULT_OUTPUT_DIR, force: bool = False
) -> tuple[Path, Path]:
    """Download the source CSV and write provenance metadata.

    Existing source data are reused unless ``force`` is true.
    """
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / DATASET_FILENAME
    metadata_path = output_dir / METADATA_FILENAME

    if data_path.exists() and not force:
        if data_path.stat().st_size == 0:
            raise ValueError(f"Existing dataset is empty: {data_path}")
        checksum = validate_source_file(data_path)
        if not metadata_path.exists():
            _write_metadata(data_path, metadata_path, checksum)
        return data_path, metadata_path

    # Stage the download beside data/raw. This prevents KaggleHub's force mode
    # from replacing unrelated files such as the provenance metadata.
    with tempfile.TemporaryDirectory(
        dir=output_dir.parent, prefix=".otdr-download-"
    ) as staging_directory:
        staging_path = Path(staging_directory).resolve()
        resolved_path = Path(
            kagglehub.dataset_download(
                DATASET_HANDLE,
                path=DATASET_FILENAME,
                output_dir=str(staging_path),
                force_download=force,
            )
        ).resolve()

        if resolved_path.is_dir():
            resolved_path = resolved_path / DATASET_FILENAME
        if not resolved_path.exists() or resolved_path.stat().st_size == 0:
            raise FileNotFoundError(
                f"Kaggle download did not produce a valid CSV: {resolved_path}"
            )
        expected_staging_path = staging_path / DATASET_FILENAME
        if resolved_path != expected_staging_path:
            raise RuntimeError(
                "Expected KaggleHub to download to "
                f"{expected_staging_path}, but it returned {resolved_path}"
            )

        checksum = validate_source_file(resolved_path)
        resolved_path.replace(data_path)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Validated source file could not be moved into place: {data_path}"
        )
    _write_metadata(data_path, metadata_path, checksum)
    return data_path, metadata_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download OTDR_data.csv from the Kaggle Optical Fibre dataset."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"destination directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing source CSV intentionally",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    data_path, metadata_path = download_dataset(args.output_dir, force=args.force)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    size_mib = int(metadata["size_bytes"]) / (1024**2)

    print(f"Dataset: {data_path}")
    print(f"Size: {size_mib:,.2f} MiB")
    print(f"SHA-256: {metadata['sha256']}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
