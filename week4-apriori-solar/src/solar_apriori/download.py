"""Download and validate the pinned Solar Power Generation dataset from Kaggle.

The downloader deliberately stages every network result outside ``data/raw``.
Only files that pass the pinned byte-size, SHA-256, and minimal CSV-schema checks
are moved into place. Kaggle credentials are handled by KaggleHub and are never
read, serialized, or printed by this module.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path

DATASET_SLUG = "anikannal/solar-power-generation-data"
DATASET_VERSION = 1
DATASET_HANDLE = f"{DATASET_SLUG}/versions/{DATASET_VERSION}"
DATASET_URL = f"https://www.kaggle.com/datasets/{DATASET_SLUG}/versions/{DATASET_VERSION}"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"
METADATA_FILENAME = "solar_power_generation.metadata.json"

GENERATION_COLUMNS = (
    "DATE_TIME",
    "PLANT_ID",
    "SOURCE_KEY",
    "DC_POWER",
    "AC_POWER",
    "DAILY_YIELD",
    "TOTAL_YIELD",
)
WEATHER_COLUMNS = (
    "DATE_TIME",
    "PLANT_ID",
    "SOURCE_KEY",
    "AMBIENT_TEMPERATURE",
    "MODULE_TEMPERATURE",
    "IRRADIATION",
)


@dataclass(frozen=True)
class FileSpec:
    """Pinned integrity and schema expectations for one source CSV."""

    size_bytes: int
    sha256: str
    required_columns: tuple[str, ...]
    plant: int
    role: str


@dataclass(frozen=True)
class ValidatedFile:
    """Observed properties of a validated source file."""

    path: Path
    size_bytes: int
    sha256: str
    columns: tuple[str, ...]


EXPECTED_FILES: Mapping[str, FileSpec] = {
    "Plant_1_Generation_Data.csv": FileSpec(
        size_bytes=4_839_076,
        sha256="882f75f1c295946633617646cc92343ddb6fb821a2546c82f7e5cc737a13624e",
        required_columns=GENERATION_COLUMNS,
        plant=1,
        role="generation",
    ),
    "Plant_1_Weather_Sensor_Data.csv": FileSpec(
        size_bytes=287_847,
        sha256="91325041328745c288cb809aebd8b1b59cde54284500902ead1fb393499d6806",
        required_columns=WEATHER_COLUMNS,
        plant=1,
        role="weather",
    ),
    "Plant_2_Generation_Data.csv": FileSpec(
        size_bytes=5_805_157,
        sha256="9b735e9278774d4d1dd4601047315ef50b3793244d892446f6aabf4fae63bd37",
        required_columns=GENERATION_COLUMNS,
        plant=2,
        role="generation",
    ),
    "Plant_2_Weather_Sensor_Data.csv": FileSpec(
        size_bytes=301_443,
        sha256="cafd80885a51b88521b3301cffa4e4f013e2516fcb3478b439a81855edac3a64",
        required_columns=WEATHER_COLUMNS,
        plant=2,
        role="weather",
    ),
}
DATASET_FILENAMES = tuple(EXPECTED_FILES)

Downloader = Callable[..., str | Path]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest without loading the whole file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_csv_schema(path: Path, required_columns: Sequence[str]) -> tuple[str, ...]:
    """Validate a CSV header and confirm the file contains at least one data row."""
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as file_handle:
            reader = csv.reader(file_handle)
            header = tuple(column.strip() for column in next(reader))
            has_data_row = next(reader, None) is not None
    except (UnicodeDecodeError, csv.Error, StopIteration) as exc:
        raise ValueError(f"Could not read a valid CSV header from {path.name}.") from exc

    if not header or any(not column for column in header):
        raise ValueError(f"CSV header is empty or malformed in {path.name}.")

    missing = [column for column in required_columns if column not in header]
    if missing:
        raise ValueError(
            f"CSV schema mismatch for {path.name}; missing columns: {', '.join(missing)}."
        )
    if not has_data_row:
        raise ValueError(f"CSV contains a header but no data rows: {path.name}.")
    return header


def validate_file(path: Path, spec: FileSpec) -> ValidatedFile:
    """Validate one pinned Kaggle v1 CSV and return its observed properties."""
    if not path.is_file():
        raise FileNotFoundError(f"Required dataset file is missing: {path.name}")

    actual_size = path.stat().st_size
    if actual_size != spec.size_bytes:
        raise ValueError(
            f"Unexpected size for {path.name}: {actual_size:,} bytes; "
            f"expected {spec.size_bytes:,} bytes for Kaggle version {DATASET_VERSION}."
        )

    actual_sha256 = sha256_file(path)
    if actual_sha256 != spec.sha256:
        raise ValueError(
            f"Unexpected SHA-256 for {path.name}: {actual_sha256}; "
            f"expected {spec.sha256}."
        )

    columns = validate_csv_schema(path, spec.required_columns)
    return ValidatedFile(
        path=path,
        size_bytes=actual_size,
        sha256=actual_sha256,
        columns=columns,
    )


def validate_dataset(directory: Path) -> dict[str, ValidatedFile]:
    """Validate all four exact source filenames in ``directory``."""
    directory = Path(directory).expanduser().resolve()
    return {
        filename: validate_file(directory / filename, spec)
        for filename, spec in EXPECTED_FILES.items()
    }


def _default_downloader(
    handle: str,
    *,
    path: str,
    output_dir: str,
    force_download: bool,
) -> str:
    """Call KaggleHub lazily so validation-only use does not require the package."""
    try:
        import kagglehub
    except ImportError:
        raise RuntimeError(
            "kagglehub is required for downloading; install the project dependencies first."
        ) from None

    return kagglehub.dataset_download(
        handle,
        path=path,
        output_dir=output_dir,
        force_download=force_download,
    )


def _package_version(distribution: str) -> str:
    try:
        return package_version(distribution)
    except PackageNotFoundError:
        return "not installed"


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _resolve_staged_file(result: str | Path, staging_dir: Path, filename: str) -> Path:
    """Resolve a downloader result while refusing files outside the staging area."""
    returned_path = Path(result).expanduser()
    candidates: list[Path] = []
    if returned_path.is_file():
        candidates.append(returned_path)
    elif returned_path.is_dir():
        candidates.append(returned_path / filename)
    candidates.append(staging_dir / filename)
    candidates.extend(staging_dir.rglob(filename))

    staging_root = staging_dir.resolve()
    unique_candidates: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
            resolved.relative_to(staging_root)
        except (OSError, ValueError):
            continue
        if resolved.is_file() and resolved not in seen:
            unique_candidates.append(resolved)
            seen.add(resolved)

    if len(unique_candidates) != 1:
        raise FileNotFoundError(
            f"Kaggle download did not produce exactly one staged {filename}."
        )
    return unique_candidates[0]


def _write_metadata(
    output_dir: Path,
    metadata_path: Path,
    validated: Mapping[str, ValidatedFile],
    acquisition: Mapping[str, str],
    downloader_name: str,
) -> None:
    files: list[dict[str, object]] = []
    for filename, spec in EXPECTED_FILES.items():
        observed = validated[filename]
        files.append(
            {
                "filename": filename,
                "local_file": _display_path(output_dir / filename),
                "plant": spec.plant,
                "role": spec.role,
                "size_bytes": observed.size_bytes,
                "sha256": observed.sha256,
                "columns": list(observed.columns),
                "acquisition_this_run": acquisition[filename],
            }
        )

    metadata: dict[str, object] = {
        "metadata_schema_version": 1,
        "dataset_slug": DATASET_SLUG,
        "dataset_handle": DATASET_HANDLE,
        "dataset_version": DATASET_VERSION,
        "dataset_url": DATASET_URL,
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "integrity": "exact byte size and SHA-256 pinned to Kaggle dataset version 1",
        "download_client": {
            "name": downloader_name,
            "kagglehub_version": _package_version("kagglehub"),
        },
        "files": files,
    }

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=metadata_path.parent,
        prefix=f".{metadata_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary_file:
        json.dump(metadata, temporary_file, indent=2)
        temporary_file.write("\n")
        temporary_path = Path(temporary_file.name)
    temporary_path.replace(metadata_path)


def download_dataset(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    force: bool = False,
    *,
    downloader: Downloader | None = None,
) -> tuple[dict[str, Path], Path]:
    """Download, verify, and atomically install the four source CSVs.

    Valid existing files are reused unless ``force`` is true. Missing or invalid
    files are downloaded into a temporary sibling directory; all staged files are
    validated before any target is replaced.
    """
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / METADATA_FILENAME
    destination_paths = {
        filename: output_dir / filename for filename in EXPECTED_FILES
    }

    validated: dict[str, ValidatedFile] = {}
    files_to_download: list[str] = []
    acquisition: dict[str, str] = {}

    for filename, spec in EXPECTED_FILES.items():
        destination = destination_paths[filename]
        if not force:
            try:
                validated[filename] = validate_file(destination, spec)
                acquisition[filename] = "reused"
                continue
            except (FileNotFoundError, OSError, ValueError):
                pass
        files_to_download.append(filename)

    download_callable = downloader or _default_downloader
    downloader_name = (
        "kagglehub"
        if downloader is None
        else f"injected:{getattr(downloader, '__name__', 'callable')}"
    )

    if files_to_download:
        with tempfile.TemporaryDirectory(
            dir=output_dir.parent,
            prefix=".solar-power-generation-download-",
        ) as temporary_directory:
            staging_dir = Path(temporary_directory).resolve()
            staged: dict[str, ValidatedFile] = {}

            for filename in files_to_download:
                try:
                    result = download_callable(
                        DATASET_HANDLE,
                        path=filename,
                        output_dir=str(staging_dir),
                        force_download=force,
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"Kaggle download failed for {filename} ({type(exc).__name__}). "
                        "Check network access and Kaggle authentication; credential values "
                        "are never logged by this command."
                    ) from None

                staged_path = _resolve_staged_file(result, staging_dir, filename)
                staged[filename] = validate_file(staged_path, EXPECTED_FILES[filename])

            # Validate every staged file before replacing any destination file.
            for filename in files_to_download:
                staged_file = staged[filename]
                staged_file.path.replace(destination_paths[filename])
                validated[filename] = ValidatedFile(
                    path=destination_paths[filename],
                    size_bytes=staged_file.size_bytes,
                    sha256=staged_file.sha256,
                    columns=staged_file.columns,
                )
                acquisition[filename] = "downloaded"

    # A final full validation also catches unexpected installation failures.
    validated = validate_dataset(output_dir)
    if files_to_download or not metadata_path.exists():
        _write_metadata(
            output_dir,
            metadata_path,
            validated,
            acquisition,
            downloader_name,
        )

    return destination_paths, metadata_path


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Download and validate Kaggle Solar Power Generation Data version 1."
        )
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
        help="redownload and replace all four source CSVs after validation",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the downloader CLI without displaying Kaggle credential material."""
    args = build_parser().parse_args(argv)
    paths, metadata_path = download_dataset(args.output_dir, force=args.force)

    print(f"Dataset: {DATASET_HANDLE}")
    for filename in EXPECTED_FILES:
        size_mib = paths[filename].stat().st_size / (1024**2)
        print(f"Validated: {paths[filename]} ({size_mib:,.2f} MiB)")
    print(f"Provenance: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
