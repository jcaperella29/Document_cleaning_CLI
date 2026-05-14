import argparse
import json
import shutil
import subprocess
from pathlib import Path


def run_command(cmd):
    print("\n▶", " ".join(cmd))
    result = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
    )

    if result.stdout:
        print(result.stdout)

    if result.stderr:
        print(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}: {' '.join(cmd)}"
        )


def load_manifest(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def summarize_manifest(engine_name, manifest):
    rows = []

    for filename, metrics in manifest.get("metrics", {}).items():
        delta = metrics.get("delta", {})

        rows.append(
            {
                "engine": engine_name,
                "file": filename,
                "selected_engine": metrics.get("selected_engine", engine_name),
                "ocr_improved": metrics.get("ocr_improved"),
                "mean_confidence_before": metrics.get("before", {}).get("mean_confidence"),
                "mean_confidence_after": metrics.get("after", {}).get("mean_confidence"),
                "delta_confidence": delta.get("mean_confidence"),
                "delta_words": delta.get("extracted_words"),
                "delta_characters": delta.get("extracted_characters"),
                "delta_low_confidence_tokens": delta.get("low_confidence_tokens"),
                "note": metrics.get("ocr_quality_note"),
            }
        )

    return rows


def write_csv(rows, output_path):
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return

    columns = [
        "engine",
        "file",
        "selected_engine",
        "ocr_improved",
        "mean_confidence_before",
        "mean_confidence_after",
        "delta_confidence",
        "delta_words",
        "delta_characters",
        "delta_low_confidence_tokens",
        "note",
    ]

    lines = [",".join(columns)]

    for row in rows:
        values = []
        for col in columns:
            value = row.get(col, "")
            value = "" if value is None else str(value)
            value = value.replace('"', '""')
            if "," in value or "\n" in value or '"' in value:
                value = f'"{value}"'
            values.append(value)
        lines.append(",".join(values))

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_summary(rows):
    print("\n==============================")
    print("BENCHMARK SUMMARY")
    print("==============================")

    if not rows:
        print("No rows found.")
        return

    by_engine = {}

    for row in rows:
        engine = row["engine"]
        by_engine.setdefault(
            engine,
            {
                "count": 0,
                "improved": 0,
                "delta_confidence": 0.0,
                "delta_words": 0,
                "delta_characters": 0,
                "delta_low_confidence_tokens": 0,
                "selected_counts": {},
            },
        )

        bucket = by_engine[engine]
        bucket["count"] += 1

        if row["ocr_improved"]:
            bucket["improved"] += 1

        bucket["delta_confidence"] += float(row["delta_confidence"] or 0)
        bucket["delta_words"] += int(row["delta_words"] or 0)
        bucket["delta_characters"] += int(row["delta_characters"] or 0)
        bucket["delta_low_confidence_tokens"] += int(
            row["delta_low_confidence_tokens"] or 0
        )

        selected = row.get("selected_engine") or engine
        bucket["selected_counts"][selected] = bucket["selected_counts"].get(selected, 0) + 1

    for engine, bucket in by_engine.items():
        count = bucket["count"] or 1

        print(f"\n{engine.upper()}")
        print(f"files: {bucket['count']}")
        print(f"ocr_improved: {bucket['improved']}/{bucket['count']}")
        print(f"avg_delta_confidence: {bucket['delta_confidence'] / count:.3f}")
        print(f"avg_delta_words: {bucket['delta_words'] / count:.3f}")
        print(f"avg_delta_characters: {bucket['delta_characters'] / count:.3f}")
        print(
            "avg_delta_low_confidence_tokens: "
            f"{bucket['delta_low_confidence_tokens'] / count:.3f}"
        )
        print(f"selected_counts: {bucket['selected_counts']}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark document cleaning engines using processor.py manifests."
    )

    parser.add_argument(
        "--weights-folder",
        default="model_weights",
        help="Folder containing CNN .mat weights.",
    )
    parser.add_argument(
        "--input-folder",
        required=True,
        help="Folder containing input images.",
    )
    parser.add_argument(
        "--output-folder",
        required=True,
        help="Folder where benchmark outputs will be written.",
    )
    parser.add_argument(
        "--engines",
        default="cnn,sbb,auto",
        help="Comma-separated engines to benchmark. Example: cnn,sbb,auto",
    )
    parser.add_argument(
        "--auto-tune",
        action="store_true",
        help="Pass --auto-tune to processor.py.",
    )
    parser.add_argument(
        "--auto-select",
        action="store_true",
        help="Pass --auto-select to processor.py.",
    )
    parser.add_argument(
        "--sbb-model-dir",
        default="external_models/sbb_binarization/saved_model",
        help="SBB model directory.",
    )
    parser.add_argument(
        "--sbb-conda-env",
        default="sbb310",
        help="Conda env containing sbb_binarize.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete output folder before running.",
    )

    args = parser.parse_args()

    output_root = Path(args.output_folder)

    if args.clean and output_root.exists():
        shutil.rmtree(output_root)

    output_root.mkdir(parents=True, exist_ok=True)

    engines = [engine.strip() for engine in args.engines.split(",") if engine.strip()]
    all_rows = []

    for engine in engines:
        engine_output = output_root / engine
        engine_output.mkdir(parents=True, exist_ok=True)

        cmd = [
            "python",
            "processor.py",
            args.weights_folder,
            args.input_folder,
            str(engine_output),
            "--engine",
            engine,
            "--sbb-model-dir",
            args.sbb_model_dir,
            "--sbb-conda-env",
            args.sbb_conda_env,
        ]

        if args.auto_tune:
            cmd.append("--auto-tune")

        if args.auto_select:
            cmd.append("--auto-select")

        run_command(cmd)

        manifest_path = engine_output / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Expected manifest not found: {manifest_path}")

        manifest = load_manifest(manifest_path)
        rows = summarize_manifest(engine, manifest)
        all_rows.extend(rows)

    summary_path = output_root / "benchmark_summary.csv"
    write_csv(all_rows, summary_path)

    print_summary(all_rows)

    print("\nSaved summary CSV:")
    print(summary_path)


if __name__ == "__main__":
    main()
