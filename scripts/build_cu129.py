"""Build the CUDA 12.9 wheels using the active Python environment."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def run(command, log, env):
    print("Running:", " ".join(map(str, command)), flush=True)
    with log.open("w") as output:
        subprocess.run(
            command, env=env, stdout=output, stderr=subprocess.STDOUT, check=True
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("build/cu129/dist"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    logs = output.parent
    env = os.environ.copy()
    env.update(
        SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG="0.5.23+cu129",
        SGLANG_BUILD_RUST_EXTS="all",
    )
    env.setdefault("CMAKE_BUILD_PARALLEL_LEVEL", "2")
    env.setdefault("CARGO_BUILD_JOBS", "2")
    env.setdefault("MAX_JOBS", "2")
    threads = int(env.get("NVCC_THREADS", "2"))
    if threads < 1:
        parser.error("NVCC_THREADS must be positive")
    env["CMAKE_ARGS"] = env.get("CMAKE_ARGS", "") + (
        f" -DSGL_KERNEL_COMPILE_THREADS={threads}"
    )
    run(
        [
            sys.executable,
            "-c",
            "import torch; "
            "assert torch.__version__ == '2.13.0+cu129'; "
            "assert torch.version.cuda == '12.9'; print(torch.__version__)",
        ],
        logs / "build-prerequisites.log",
        env,
    )
    packages = [root / "python/sglang/kernels/aot", root / "python"]
    wheels = []
    for name, package in zip(("sglang_kernel", "sglang"), packages):
        command = [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(output),
        ]
        if name == "sglang_kernel":
            command += ["-Cbuild-dir=" + str(logs / "kernel-build")]
        run(command + [str(package)], logs / f"{name}-build.log", env)
        wheel = next(output.glob(f"{name}-*+cu129-*.whl"))
        if name == "sglang_kernel":
            # infllm_ops uses pybind11's version-specific Python ABI even though
            # the other extensions use the stable ABI. Tag the whole wheel safely.
            python_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
            run(
                [
                    sys.executable,
                    "-m",
                    "wheel",
                    "tags",
                    "--python-tag",
                    python_tag,
                    "--abi-tag",
                    python_tag,
                    "--remove",
                    str(wheel),
                ],
                logs / "kernel-wheel-tags.log",
                env,
            )
            wheel = next(output.glob(f"{name}-*+cu129-{python_tag}-{python_tag}-*.whl"))
        # Build-directory RPATHs must not leak into the release artifacts.
        with tempfile.TemporaryDirectory(dir=logs) as temporary:
            directory = Path(temporary)
            run(
                [sys.executable, "-m", "wheel", "unpack", str(wheel), "-d", temporary],
                logs / "wheel-unpack.log",
                env,
            )
            unpacked = next(directory.iterdir())
            for library in unpacked.rglob("*.so"):
                rpath = subprocess.check_output(
                    ["patchelf", "--print-rpath", str(library)], env=env, text=True
                ).strip()
                relative = ":".join(
                    p for p in rpath.split(":") if p and not p.startswith("/")
                )
                subprocess.run(
                    ["patchelf", "--set-rpath", relative, str(library)],
                    env=env,
                    check=True,
                )
            run(
                [sys.executable, "-m", "wheel", "pack", str(unpacked), "-d", temporary],
                logs / "wheel-pack.log",
                env,
            )
            next(directory.glob("*.whl")).replace(wheel)
        wheels.append(wheel)
    record = {
        "source_commit": subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip(),
        "python": sys.version,
        "source_diff_sha256": hashlib.sha256(
            subprocess.check_output(["git", "-C", str(root), "diff", "HEAD"])
        ).hexdigest(),
        "torch": "2.13.0+cu129",
        "cuda_architectures": "upstream CUDA 12.9 defaults",
        "wheels": {
            p.name: hashlib.file_digest(p.open("rb"), "sha256").hexdigest()
            for p in wheels
        },
    }
    (output / "build-manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    (output / "SHA256SUMS").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in record["wheels"].items())
    )
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
