"""Measure CPU inference latency and FPS for YOLOv4-Tiny (and optionally full
YOLOv4) under one or more edge profiles. Writes results/benchmark.json and a
bar chart results/fps_plot.png.

Every number here is measured on the host CPU restricted to each profile's
thread budget. This is an edge-simulated profile, not a real Raspberry Pi.

Example:
    python src/benchmark.py --models tiny --profiles unconstrained,pi4b,edge_2core
"""

import argparse
import json
import os
import platform
import statistics
import subprocess
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import edge_sim  # noqa: E402
from detect import build_net, infer  # noqa: E402
from utils import output_layer_names, rel  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "data/samples"


def parse_args():
    p = argparse.ArgumentParser(description="YOLOv4-Tiny CPU benchmark")
    p.add_argument("--models", default="tiny", help="comma list: tiny,full")
    p.add_argument("--profiles", default="unconstrained,pi4b,pi4b_320,edge_2core",
                   help="comma list of edge profile names")
    p.add_argument("--image", default=None,
                   help="image to benchmark on (default: first in data/samples)")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--nms", type=float, default=0.45)
    p.add_argument("--output", default=str(ROOT / "results"))
    p.add_argument("--label", default="",
                   help="suffix for output files, e.g. --label full -> benchmark_full.json")
    return p.parse_args()


def pick_image(arg):
    """Use the image the caller named, or fall back to the first sample."""
    if arg:
        return Path(arg)
    images = sorted(p for p in SAMPLES.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not images:
        raise SystemExit("no sample image found; pass --image or add files to data/samples")
    return images[0]


def cpu_model():
    """Best-effort CPU name for the report: /proc on Linux, sysctl on macOS,
    and a generic platform string anywhere else."""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    try:
        out = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"],
                                      stderr=subprocess.DEVNULL, text=True).strip()
        if out:
            return out
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or platform.machine()


def time_run(net, out_names, frame, input_size, conf, nms, warmup, iters):
    """Throw away a few warmup passes so caches and threads settle, then time the
    real iterations and hand back the per-iteration latencies in milliseconds."""
    for _ in range(warmup):
        infer(net, out_names, frame, input_size, conf, nms)
    latencies = []
    for _ in range(iters):
        _, latency = infer(net, out_names, frame, input_size, conf, nms)
        latencies.append(latency)
    return latencies


def stats(latencies):
    """Reduce a list of latencies to mean, median, p95 and the matching FPS."""
    ordered = sorted(latencies)
    p95 = ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]
    mean = statistics.mean(latencies)
    return {
        "mean_ms": round(mean, 2),
        "median_ms": round(statistics.median(latencies), 2),
        "p95_ms": round(p95, 2),
        "fps": round(1000.0 / mean, 2),
    }


def make_plot(runs, models, profiles, path):
    """Draw the FPS bar chart, one cluster per profile and one bar per model."""
    x = np.arange(len(profiles))
    width = 0.8 / max(1, len(models))
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, model in enumerate(models):
        fps = []
        for prof in profiles:
            match = next((r for r in runs if r["model"] == model and r["profile"] == prof), None)
            fps.append(match["fps"] if match else 0)
        bars = ax.bar(x + i * width, fps, width, label=f"yolov4-{model}")
        ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=8)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(profiles, rotation=15)
    ax.set_ylabel("FPS (CPU, edge-simulated)")
    ax.set_title("CPU inference FPS by edge profile")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    args = parse_args()
    image_path = pick_image(args.image)
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise SystemExit(f"could not read image {image_path}")

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    profiles = [p.strip() for p in args.profiles.split(",") if p.strip()]

    print(f"image {image_path.name} {frame.shape[1]}x{frame.shape[0]}, "
          f"{args.warmup} warmup + {args.iters} timed iters, CPU-only\n")
    runs = []
    for model in models:
        net = build_net(model)
        out_names = output_layer_names(net)
        for name in profiles:
            profile = edge_sim.get_profile(name)
            input_size = profile.get("input_size", 416)
            with edge_sim.apply_profile(profile):
                latencies = time_run(net, out_names, frame, input_size,
                                     args.conf, args.nms, args.warmup, args.iters)
            row = {"model": model, "profile": name,
                   "threads": int(profile.get("threads", 0) or 0),
                   "input_size": input_size, **stats(latencies)}
            runs.append(row)
            print(f"{model:5s} {name:14s} {row['mean_ms']:7.1f} ms mean  "
                  f"{row['median_ms']:7.1f} ms med  {row['p95_ms']:7.1f} ms p95  "
                  f"{row['fps']:6.1f} FPS")

    report = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "note": "edge-simulated on host CPU restricted per profile; not a Raspberry Pi",
        "host": {
            "platform": platform.platform(),
            "cpu": cpu_model(),
            "cpu_count": os.cpu_count(),
            "opencv": cv2.__version__,
        },
        "params": {"image": image_path.name, "warmup": args.warmup,
                   "iters": args.iters, "conf": args.conf, "nms": args.nms},
        "runs": runs,
    }
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.label}" if args.label else ""
    json_path = out_dir / f"benchmark{suffix}.json"
    plot_path = out_dir / f"fps_plot{suffix}.png"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {rel(json_path, ROOT)}")

    make_plot(runs, models, profiles, plot_path)
    print(f"wrote {rel(plot_path, ROOT)}")


if __name__ == "__main__":
    main()
