import os
import subprocess
from pathlib import Path


def run_sbb_binarization(
    input_path: str,
    output_path: str,
    model_dir: str,
    conda_env: str = "sbb310",
) -> str:
    """
    Run SBB binarization through a pinned conda env.

    SBB needs an older TensorFlow/Keras stack, so we intentionally run it
    as a subprocess instead of importing it into the main app.
    """
    input_file = Path(input_path)
    output_file = Path(output_path)
    model_path = Path(model_dir)

    if not input_file.exists():
        raise FileNotFoundError(f"Input image not found: {input_file}")

    if not model_path.exists():
        raise FileNotFoundError(f"SBB model directory not found: {model_path}")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "conda",
        "run",
        "-n",
        conda_env,
        "sbb_binarize",
        "-m",
        str(model_path),
        str(input_file),
        str(output_file),
    ]

    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"

    result = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "SBB binarization failed.\n"
            f"Command: {' '.join(cmd)}\n\n"
            f"STDOUT:\n{result.stdout}\n\n"
            f"STDERR:\n{result.stderr}"
        )

    if not output_file.exists():
        raise RuntimeError(
            "SBB completed but did not create an output file.\n"
            f"Expected: {output_file}\n"
            f"STDOUT:\n{result.stdout}\n\n"
            f"STDERR:\n{result.stderr}"
        )

    return str(output_file)
