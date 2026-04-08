from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import modal

APP_NAME = "tensorcore-gemm-runner"
LOCAL_REPO_DIR = Path(__file__).resolve().parent
REMOTE_SOURCE_DIR = Path("/root/local-src/tensorcore-gemm")
DEFAULT_GPU = "L4"
RESULTS_DIR = "results"

app = modal.App(APP_NAME)

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.11")
    .entrypoint([])
    .apt_install("git")
    .pip_install("uv", "setuptools", "ninja")
    .add_local_dir(
        LOCAL_REPO_DIR,
        remote_path=str(REMOTE_SOURCE_DIR),
        copy=True,
        ignore=[".git", "__pycache__", "*.pyc", ".venv", ".modal-venv", ".DS_Store", "results"],
    )
    .workdir(str(REMOTE_SOURCE_DIR))
    .run_commands("uv sync --extra cuda --extra bench")
    .run_commands("git clone --depth 1 https://github.com/meta-pytorch/tritonbench.git /opt/tritonbench")
    .run_commands("uv pip install --python .venv/bin/python --no-deps -e /opt/tritonbench")
)


def _run(command: list[str]) -> str:
    result = subprocess.run(command, cwd=str(REMOTE_SOURCE_DIR), text=True, capture_output=True)
    output = []
    if result.stdout.strip():
        output.append(result.stdout.strip())
    if result.stderr.strip():
        output.append(result.stderr.strip())
    combined = "\n".join(output)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(command)}\n{combined}")
    return combined


@app.function(
    image=image,
    gpu=DEFAULT_GPU,
    cpu=8,
    memory=32768,
    timeout=60 * 60 * 2,
)
def run_benchmark(
    script: str = "benchmark.py",
    m: int = 4096,
    n: int = 4096,
    k: int = 4096,
    cases: str = "",
    warmup: int = 25,
    iters: int = 100,
    modes: str = "wmma,mma",
) -> dict[str, str]:
    result_name = "latest_l4.json" if script != "benchmark_tritonbench.py" else "latest_l4_tritonbench.json"
    stdout = _run(
        [
            "uv",
            "run",
            "python",
            script,
            "--m",
            str(m),
            "--n",
            str(n),
            "--k",
            str(k),
            "--cases",
            cases,
            "--warmup",
            str(warmup),
            "--iters",
            str(iters),
            "--modes",
            modes,
            "--json-out",
            f"{RESULTS_DIR}/{result_name}",
        ],
    )
    artifacts: dict[str, str] = {"stdout.txt": stdout}
    result_path = REMOTE_SOURCE_DIR / RESULTS_DIR / result_name
    if result_path.exists():
        artifacts[result_name] = result_path.read_text()
    return artifacts


@app.local_entrypoint()
def main(
    action: str = "benchmark",
    script: str = "benchmark.py",
    m: int = 4096,
    n: int = 4096,
    k: int = 4096,
    cases: str = "",
    warmup: int = 25,
    iters: int = 100,
    modes: str = "wmma,mma",
) -> None:
    if action not in {"benchmark", "tritonbench"}:
        raise ValueError("Supported actions: benchmark, tritonbench")

    if action == "tritonbench" and script == "benchmark.py":
        script = "benchmark_tritonbench.py"

    artifacts = run_benchmark.remote(
        script=script,
        m=m,
        n=n,
        k=k,
        cases=cases,
        warmup=warmup,
        iters=iters,
        modes=modes,
    )

    stdout = artifacts.get("stdout.txt", "")
    if stdout:
        print(stdout)

    output_dir = LOCAL_REPO_DIR / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename_prefix = "l4-tritonbench" if action == "tritonbench" else "l4-benchmark"
    result_name = "latest_l4_tritonbench.json" if action == "tritonbench" else "latest_l4.json"
    output_path = output_dir / f"{filename_prefix}-{timestamp}.json"
    result_json = artifacts.get(result_name)
    if not result_json:
        raise RuntimeError(f"Modal benchmark completed without returning results/{result_name}")
    output_path.write_text(json.dumps(json.loads(result_json), indent=2))
    print(f"\nSaved local results to {output_path}")
