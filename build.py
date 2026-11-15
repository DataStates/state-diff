#!/usr/bin/env python3
"""
build.py -- Automated build script for state-diff.

Builds state-diff and its direct dependency scissor-io if it is not already
installed.  The full build order is:

  1. scissor-io     built with plain cmake from the sibling directory
                    ../scissor-io, unless --scissor-prefix points to a
                    pre-installed copy.
  2. state-diff

Usage:
  python3 build.py [options]

Options:
  --prefix DIR           Install prefix for state-diff
                         (default: <state-diff_root>/_install)
  --build-dir DIR        Root for build trees
                         (default: <state-diff_root>/_build)
  --scissor-prefix DIR   Path to an already-installed scissor-io.
                         If omitted, scissor-io is built from source (expected
                         as a sibling directory: ../scissor-io).
  --kokkos-dir DIR       Path to the Kokkos CMake config directory
                         (e.g. /path/to/kokkos/lib/cmake/Kokkos).
                         Passed as -DKokkos_DIR to CMake.
  --build-type TYPE      CMake build type: Debug | Release | RelWithDebInfo
                         (default: Release)
  --jobs N               Parallel build jobs (default: number of CPU cores)
  --no-tests             Skip building and running tests
  --clean                Delete existing build directories before configuring
  -v / --verbose         Show full cmake and build output
  -h / --help            Show this help message

Example (build everything from source):
  python3 build.py --kokkos-dir /path/to/kokkos/lib/cmake/Kokkos

Example (use a pre-installed scissor-io):
  python3 build.py \\
      --scissor-prefix /opt/scissor \\
      --kokkos-dir /opt/kokkos/lib/cmake/Kokkos

Example (production install):
  python3 build.py \\
      --prefix /opt/statediff \\
      --scissor-prefix /opt/scissor \\
      --kokkos-dir /opt/kokkos/lib/cmake/Kokkos \\
      --build-type RelWithDebInfo
"""

import argparse
import multiprocessing
import os
import shutil
import subprocess
import sys
from pathlib import Path


# ======================================================
# Helpers
# ======================================================

def log(msg: str) -> None:
    print(f"[build.py] {msg}", flush=True)


def die(msg: str) -> None:
    print(f"[build.py] ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def run(cmd: list[str], env: dict | None = None, verbose: bool = False,
        cwd: Path | None = None) -> None:
    """Run a command, streaming output only if verbose or on failure."""
    display = " ".join(str(c) for c in cmd)
    log(f"$ {display}")
    result = subprocess.run(
        cmd,
        env=env,
        cwd=cwd,
        capture_output=not verbose,
        text=True,
    )
    if result.returncode != 0:
        if not verbose:
            sys.stdout.write(result.stdout)
            sys.stderr.write(result.stderr)
        die(f"Command failed (exit {result.returncode}): {display}")


def cmake_configure(src: Path, build: Path, install: Path, options: dict,
                    env: dict | None = None, verbose: bool = False) -> None:
    cmd = [
        "cmake",
        "-S", str(src),
        "-B", str(build),
        f"-DCMAKE_INSTALL_PREFIX={install}",
    ]
    for key, val in options.items():
        cmd.append(f"-D{key}={val}")
    run(cmd, env=env, verbose=verbose)


def cmake_build(build: Path, jobs: int, verbose: bool = False,
                env: dict | None = None) -> None:
    cmd = ["cmake", "--build", str(build), f"-j{jobs}"]
    if verbose:
        cmd += ["--", "VERBOSE=1"]
    run(cmd, env=env, verbose=verbose)


def cmake_install(build: Path, verbose: bool = False,
                  env: dict | None = None) -> None:
    run(["cmake", "--install", str(build)], env=env, verbose=verbose)


def cmake_test(build: Path, verbose: bool = False,
               env: dict | None = None) -> None:
    cmd = ["ctest", "--output-on-failure", "--timeout", "120"]
    if verbose:
        cmd.append("-V")
    run(cmd, cwd=build, env=env, verbose=verbose)


# ======================================================
# Build steps
# ======================================================

def build_scissor(statediff_root: Path, build_root: Path,
                  install_prefix: Path,
                  args: argparse.Namespace) -> Path:
    """Build and install scissor-io from ../scissor-io with plain cmake.
    Returns the install prefix used for scissor-io."""
    scissor_src = statediff_root.parent / "scissor-io"

    if not (scissor_src / "CMakeLists.txt").is_file():
        die(f"scissor-io sources not found at {scissor_src}.  "
            "Pass --scissor-prefix to point at a pre-installed scissor-io.")

    # Install scissor-io into its own subdirectory so state-diff's
    # install prefix stays clean.
    scissor_install = install_prefix.parent / (install_prefix.name + "-scissor")
    scissor_build   = build_root / "scissor-io"

    if args.clean and scissor_build.exists():
        log(f"Removing {scissor_build}")
        shutil.rmtree(scissor_build)
    scissor_build.mkdir(parents=True, exist_ok=True)

    options = {
        "CMAKE_BUILD_TYPE":     args.build_type,
        "SCISSOR_ENABLE_TESTS": "OFF",   # avoid re-running scissor tests here
        "SCISSOR_ENABLE_TOOLS": "OFF",
    }

    log("=== Building scissor-io ===")
    cmake_configure(scissor_src, scissor_build, scissor_install, options,
                    verbose=args.verbose)
    cmake_build(scissor_build, args.jobs, verbose=args.verbose)
    cmake_install(scissor_build, verbose=args.verbose)
    log("scissor-io installed successfully.")
    return scissor_install


def build_statediff(statediff_root: Path, build_root: Path,
                    install_prefix: Path, scissor_prefix: Path,
                    args: argparse.Namespace) -> None:
    src   = statediff_root
    build = build_root / "state-diff"

    if args.clean and build.exists():
        log(f"Removing {build}")
        shutil.rmtree(build)
    build.mkdir(parents=True, exist_ok=True)

    options = {
        "CMAKE_BUILD_TYPE":  args.build_type,
        "CMAKE_PREFIX_PATH": str(scissor_prefix),
    }
    if args.kokkos_dir:
        options["Kokkos_DIR"] = args.kokkos_dir

    log("=== Configuring state-diff ===")
    cmake_configure(src, build, install_prefix, options, verbose=args.verbose)

    log("=== Building state-diff ===")
    cmake_build(build, args.jobs, verbose=args.verbose)

    if not args.no_tests:
        log("=== Running state-diff tests ===")
        cmake_test(build, verbose=args.verbose)

    log("=== Installing state-diff ===")
    cmake_install(build, verbose=args.verbose)

    log("state-diff installed successfully.")


# ======================================================
# Argument parsing
# ======================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--prefix", metavar="DIR", default=None,
        help="Install prefix for state-diff (default: <state-diff_root>/_install)",
    )
    p.add_argument(
        "--build-dir", metavar="DIR", default=None,
        help="Root for build trees (default: <state-diff_root>/_build)",
    )
    p.add_argument(
        "--scissor-prefix", metavar="DIR", default=None,
        help="Path to a pre-installed scissor-io.  "
             "If omitted, scissor-io is built from ../scissor-io.",
    )
    p.add_argument(
        "--kokkos-dir", metavar="DIR", default=None,
        help="Path to the Kokkos CMake config dir (e.g. .../lib/cmake/Kokkos)",
    )
    p.add_argument(
        "--build-type", metavar="TYPE", default="Release",
        choices=["Debug", "Release", "RelWithDebInfo", "MinSizeRel"],
        help="CMake build type (default: Release)",
    )
    p.add_argument(
        "--jobs", "-j", type=int,
        default=multiprocessing.cpu_count(),
        help="Parallel build jobs (default: all CPU cores)",
    )
    p.add_argument(
        "--no-tests", action="store_true",
        help="Skip building and running tests",
    )
    p.add_argument(
        "--clean", action="store_true",
        help="Delete existing build directories before configuring",
    )
    p.add_argument(
        "-v", "--verbose", action="store_true",
        help="Show full cmake and build output",
    )
    return p.parse_args()


# ======================================================
# Main
# ======================================================

def main() -> None:
    args = parse_args()

    # This script lives in the state-diff source directory
    statediff_root = Path(__file__).resolve().parent

    # Default build root is a sibling of the source directory (not inside it)
    # to avoid CMake INTERFACE_INCLUDE_DIRECTORIES restrictions from ExternalProject.
    build_root     = Path(args.build_dir).resolve()  if args.build_dir else statediff_root.parent / "_build" / statediff_root.name
    install_prefix = Path(args.prefix).resolve()     if args.prefix    else statediff_root / "_install"

    log(f"state-diff root: {statediff_root}")
    log(f"Build root     : {build_root}")
    log(f"Install prefix : {install_prefix}")
    log(f"Build type     : {args.build_type}")
    log(f"Jobs           : {args.jobs}")

    # Resolve the scissor install prefix
    if args.scissor_prefix:
        scissor_prefix = Path(args.scissor_prefix).resolve()
        log(f"Scissor prefix : {scissor_prefix} (pre-installed)")
    else:
        scissor_prefix = build_scissor(statediff_root, build_root,
                                       install_prefix, args)
        log(f"Scissor prefix : {scissor_prefix} (built from source)")

    # Build state-diff
    build_statediff(statediff_root, build_root, install_prefix,
                    scissor_prefix, args)

    log("")
    log("=== Build complete ===")
    log(f"Headers  : {install_prefix}/include/")
    log(f"Libraries: {install_prefix}/lib/")
    log(f"CMake    : {install_prefix}/lib/cmake/statediff/")
    log(f"CLI      : {install_prefix}/bin/statediff-cli")
    log("")
    log("To use in a downstream CMake project:")
    log(f"  cmake ... -DCMAKE_PREFIX_PATH={install_prefix}")


if __name__ == "__main__":
    main()
