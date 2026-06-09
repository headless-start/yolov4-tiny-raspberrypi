"""Make a fast dev CPU behave a bit more like a small edge device.

The profiles live in configs/edge_profiles.yaml. From Python we can only really
control two things, so that is what this module does: the thread count (through
cv2.setNumThreads and the matching OMP/OpenBLAS env vars) and forcing inference
onto the CPU, since a Pi has no CUDA. The rest of a profile, the core and memory
budget, is something you apply from the outside with taskset/cpulimit or
docker --cpus/--memory; the README spells those out.
"""

import contextlib
import os
from pathlib import Path

import cv2
import yaml

PROFILE_PATH = Path(__file__).resolve().parent.parent / "configs" / "edge_profiles.yaml"

_THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
               "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def load_profiles(path=PROFILE_PATH):
    with open(path) as f:
        return yaml.safe_load(f)


def get_profile(name, path=PROFILE_PATH):
    """Look up one profile by name, with a helpful error if it isn't defined."""
    profiles = load_profiles(path)
    if name not in profiles:
        raise KeyError(f"unknown profile '{name}'; available: {', '.join(profiles)}")
    return profiles[name]


def cpu_target(net):
    """Pin a loaded net to the OpenCV CPU backend (a Pi has no CUDA)."""
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net


def describe(profile, name=""):
    """One readable line about the active profile, printed so every result is
    labelled with the envelope it was produced under."""
    threads = int(profile.get("threads", 0) or 0)
    thr = "all cores" if threads == 0 else f"{threads} thread(s)"
    size = profile.get("input_size", 416)
    line = f"edge-sim profile '{name}': {thr}, CPU-only, {size}x{size} input"
    envelope = []
    if profile.get("cpus"):
        envelope.append(f"{profile['cpus']} cpus")
    if profile.get("memory"):
        envelope.append(str(profile["memory"]))
    if envelope:
        line += f"  (imitates {', '.join(envelope)})"
    return line


@contextlib.contextmanager
def apply_profile(profile):
    """Hold OpenCV to the profile's thread budget inside a `with` block, then put
    the old setting back on the way out. threads=0 in a profile means leave every
    core available.
    """
    threads = int(profile.get("threads", 0) or 0)
    if threads <= 0:
        threads = os.cpu_count() or 1
    previous = cv2.getNumThreads()
    cv2.setNumThreads(threads)
    for var in _THREAD_ENV:
        os.environ[var] = str(threads)
    try:
        yield profile
    finally:
        cv2.setNumThreads(previous)
