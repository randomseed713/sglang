"""Build the CUDA 12.9 wheels using the active Python environment."""

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import sysconfig
import tarfile
import tempfile
from pathlib import Path


def run(command, log, env):
    print("Running:", " ".join(map(str, command)), flush=True)
    with log.open("w") as output:
        subprocess.run(
            command, env=env, stdout=output, stderr=subprocess.STDOUT, check=True
        )


def kernel_graph(root, directory, env, log):
    """Configure the same upstream targets without running the wheel installer."""
    import torch
    from torch.utils import cmake_prefix_path

    assert torch.__version__ == "2.13.0+cu129", torch.__version__
    assert torch.version.cuda == "12.9", torch.version.cuda
    command = [
        "cmake",
        "-S",
        str(root / "python/sglang/kernels/aot"),
        "-B",
        str(directory),
        "-G",
        "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DPython_EXECUTABLE={sys.executable}",
        f"-DPython3_EXECUTABLE={sys.executable}",
        f"-DPython_INCLUDE_DIR={sysconfig.get_path('include')}",
        f"-DPython3_INCLUDE_DIR={sysconfig.get_path('include')}",
        "-DSKBUILD_SABI_VERSION=3.11",
        "-DSKBUILD_SABI_COMPONENT=Development.SABIModule",
        f"-DCMAKE_PREFIX_PATH={cmake_prefix_path}",
    ] + shlex.split(env["CMAKE_ARGS"])
    run(command, log, env)
    database = json.loads(
        subprocess.check_output(
            ["ninja", "-C", str(directory), "-t", "compdb", "-x"], env=env, text=True
        )
    )
    return {item["output"]: item["command"] for item in database if item["command"]}


def checksum(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def kernel_objects(root, directory, env, logs, index, count):
    graph = kernel_graph(root, directory, env, logs / "kernel-configure.log")
    objects = sorted(name for name in graph if name.endswith(".o"))[index::count]
    if not objects:
        raise ValueError("Kernel shard contains no objects")
    run(
        [
            "ninja",
            "-C",
            str(directory),
            "-j",
            env["CMAKE_BUILD_PARALLEL_LEVEL"],
            *objects,
        ],
        logs / "kernel-objects.log",
        env,
    )
    manifest = {
        "commit": subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip(),
        "python": list(sys.version_info[:2]),
        "index": index,
        "count": count,
        "objects": {
            name: {
                "sha256": checksum(directory / name),
                "command_sha256": hashlib.sha256(graph[name].encode()).hexdigest(),
            }
            for name in objects
        },
    }
    archive = logs / f"kernel-objects-{index}.tar.gz"
    with tempfile.TemporaryDirectory(dir=logs) as temporary:
        metadata = Path(temporary) / "manifest.json"
        metadata.write_text(json.dumps(manifest))
        with tarfile.open(archive, "w:gz", compresslevel=1) as tar:
            tar.add(metadata, arcname="manifest.json")
            for name in objects:
                tar.add(directory / name, arcname=f"objects/{name}")
    print(f"Built {len(objects)} objects: {archive}", flush=True)


def assemble_kernel(root, directory, env, logs, archives, count):
    graph = kernel_graph(root, directory, env, logs / "kernel-configure.log")
    expected = {name for name in graph if name.endswith(".o")}
    commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    seen, indices = set(), set()
    for archive in sorted(archives.rglob("kernel-objects-*.tar.gz")):
        with tarfile.open(archive) as tar:
            manifest = json.load(tar.extractfile("manifest.json"))
            assert manifest["commit"] == commit, archive
            assert manifest["python"] == list(sys.version_info[:2]), archive
            assert manifest["count"] == count, archive
            assert manifest["index"] not in indices, archive
            indices.add(manifest["index"])
            for name, record in manifest["objects"].items():
                assert name in expected and name not in seen, name
                assert (
                    record["command_sha256"]
                    == hashlib.sha256(graph[name].encode()).hexdigest()
                ), name
                destination = directory / name
                assert destination.resolve().is_relative_to(directory.resolve()), name
                destination.parent.mkdir(parents=True, exist_ok=True)
                member = tar.getmember(f"objects/{name}")
                assert member.isfile(), name
                with (
                    tar.extractfile(member) as source,
                    destination.open("wb") as target,
                ):
                    shutil.copyfileobj(source, target)
                assert checksum(destination) == record["sha256"], name
                seen.add(name)
    assert indices == set(range(count)), indices
    assert seen == expected, expected - seen
    # Link exactly the upstream commands after verifying every object's source
    # commit, Python version, compiler command and content checksum.
    with (logs / "kernel-link.log").open("w") as output:
        for name, command in graph.items():
            if name.endswith(".so"):
                (directory / name).parent.mkdir(parents=True, exist_ok=True)
                subprocess.run(
                    command,
                    cwd=directory,
                    env=env,
                    shell=True,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    check=True,
                )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("build/cu129/dist"))
    parser.add_argument("--kernel-shard", type=int)
    parser.add_argument("--kernel-shards", type=int, default=8)
    parser.add_argument("--assemble-kernel", type=Path)
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
    directory = logs / "kernel-build"
    if args.kernel_shard is not None:
        if not 0 <= args.kernel_shard < args.kernel_shards:
            parser.error("Invalid kernel shard index")
        kernel_objects(
            root, directory, env, logs, args.kernel_shard, args.kernel_shards
        )
        return
    if args.assemble_kernel:
        assemble_kernel(
            root,
            directory,
            env,
            logs,
            args.assemble_kernel.resolve(),
            args.kernel_shards,
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
            command += ["-Cbuild-dir=" + str(directory)]
            if args.assemble_kernel:
                # Objects and libraries were built above. Let the normal wheel
                # backend configure/install/package without compiling them again.
                command += ["-Cbuild.tool-args=-n"]
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
