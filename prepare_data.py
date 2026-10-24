import argparse
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

from data import prepare_data as build_cache, sha256, unpack_prepared


def create_archive(prepared, assets, destination):
    identity = json.loads((assets / "identity.json").read_text())
    records = {name: dict(bytes=(prepared / name).stat().st_size, sha256=digest)
               for name, digest in identity["cache_sha256"].items()}
    records["tokenizer.json"] = dict(bytes=(assets / "tokenizer.json").stat().st_size,
                                    sha256=identity["tokenizer_sha256"])
    shutil.copy2(assets / "tokenizer.json", prepared / "tokenizer.json")
    manifest = dict(kind="student_prepared_data_v1", train_rows=identity["train_rows"],
                    validation_rows=identity["validation_rows"], files=records)
    (prepared / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    temporary = destination.with_suffix(".tmp.zip")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in sorted(prepared.rglob("*")):
                if not path.is_file():
                    continue
                name = "colab_training_data/" + path.relative_to(prepared).as_posix()
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                with path.open("rb") as src, archive.open(info, "w", force_zip64=True) as dst:
                    shutil.copyfileobj(src, dst, length=8 * 1024**2)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    assets = Path(__file__).resolve().parent / "data"
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=assets / "raw_dataset/data/train")
    parser.add_argument("--output", type=Path, default=assets / "prepared_data.zip")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="prepare_", dir=args.output.parent) as temporary:
        prepared = Path(temporary) / "colab_training_data"
        if args.output.exists():
            unpack_prepared(assets, archive=args.output, output=prepared)
        else:
            build_cache(args.raw_dir, prepared, assets)
            create_archive(prepared, assets, args.output)
    print(json.dumps(dict(archive=str(args.output), sha256=sha256(args.output),
                          train_rows=5820047, validation_rows=1500), indent=2))


if __name__ == "__main__":
    main()
