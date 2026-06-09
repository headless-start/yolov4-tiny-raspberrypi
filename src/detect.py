"""Run YOLOv4-Tiny (or full YOLOv4) detection on an image, folder, video or webcam.

Inference is CPU-only via OpenCV's DNN module, optionally throttled to a named
edge profile so the speed matches a constrained device.

Examples:
    python src/detect.py --source data/samples/dog.jpg
    python src/detect.py --source data/samples --profile pi4b
    python src/detect.py --source clip.mp4 --model tiny --profile pi4b_320
    python src/detect.py --source 0 --show          # webcam
"""

import argparse
import time
from collections import Counter
from pathlib import Path

import cv2

import edge_sim
from utils import (FpsMeter, draw_detections, letterbox, load_class_names,
                   output_layer_names, postprocess, rel)

ROOT = Path(__file__).resolve().parent.parent
MODELS = {
    "tiny": (ROOT / "models/yolov4-tiny.cfg", ROOT / "models/yolov4-tiny.weights"),
    "full": (ROOT / "models/yolov4.cfg", ROOT / "models/yolov4.weights"),
}
NAMES = ROOT / "models/coco.names"
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def gui_available():
    """True if this OpenCV build can open windows; the headless wheel cannot."""
    try:
        cv2.namedWindow("__probe__")
        cv2.destroyWindow("__probe__")
        return True
    except cv2.error:
        return False


def build_net(model):
    """Load the chosen darknet weights and wire the net up to run on the CPU."""
    cfg, weights = MODELS[model]
    if not cfg.exists() or not weights.exists():
        raise SystemExit(f"missing {model} model files; run 'python models/download_weights.py' first")
    net = cv2.dnn.readNetFromDarknet(str(cfg), str(weights))
    return edge_sim.cpu_target(net)


def infer(net, out_names, frame, input_size, conf, nms):
    """Push one frame through the net and return the detections along with how
    many milliseconds the whole step took, letterbox and decode included."""
    t0 = time.perf_counter()
    padded, scale, pad_w, pad_h = letterbox(frame, input_size)
    blob = cv2.dnn.blobFromImage(padded, 1 / 255.0, (input_size, input_size),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(out_names)
    detections = postprocess(outputs, input_size, scale, pad_w, pad_h, conf, nms)
    latency = (time.perf_counter() - t0) * 1000
    return detections, latency


def summarise(detections, class_names):
    """A short '2x person, 1x dog' line describing what turned up in a frame."""
    counts = Counter(class_names[c] for c, _, _ in detections)
    return ", ".join(f"{n}x {k}" for k, n in counts.items()) or "nothing"


def run_images(net, out_names, paths, class_names, args, profile, out_dir):
    """Detect on a set of still images, save each annotated result, and print the
    count and latency for each one."""
    input_size = profile.get("input_size", 416)
    for path in paths:
        frame = cv2.imread(str(path))
        if frame is None:
            print(f"skip {path} (could not read)")
            continue
        detections, latency = infer(net, out_names, frame, input_size, args.conf, args.nms)
        annotated = draw_detections(frame, detections, class_names)
        dest = out_dir / f"{path.stem}_{args.model}.jpg"
        cv2.imwrite(str(dest), annotated)
        print(f"{path.name}: {len(detections)} objects ({summarise(detections, class_names)}) "
              f"in {latency:.1f} ms -> {rel(dest, ROOT)}")


def run_stream(net, out_names, source, class_names, args, profile, out_dir):
    """Detect on a video file or webcam one frame at a time, overlay a live FPS,
    and write an annotated video back out for file sources."""
    input_size = profile.get("input_size", 416)
    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else str(source))
    if not cap.isOpened():
        raise SystemExit(f"could not open source {source}")

    writer = None
    if not str(source).isdigit():
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        dest = out_dir / f"{Path(str(source)).stem}_{args.model}.mp4"
        writer = cv2.VideoWriter(str(dest), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    meter = FpsMeter()
    frames = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            detections, latency = infer(net, out_names, frame, input_size, args.conf, args.nms)
            annotated = draw_detections(frame, detections, class_names)
            fps = meter.update(latency / 1000)
            cv2.putText(annotated, f"{fps:.1f} FPS", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
            if writer is not None:
                writer.write(annotated)
            if args.show:
                cv2.imshow("detect", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            frames += 1
            print(f"frame {frames}: {latency:.1f} ms, {fps:.1f} FPS, {len(detections)} objects")
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if writer is not None:
            writer.release()
            print(f"wrote {rel(dest, ROOT)}")
        if args.show:
            cv2.destroyAllWindows()


def parse_args():
    p = argparse.ArgumentParser(description="YOLOv4-Tiny CPU detection")
    p.add_argument("--source", required=True, help="image, folder, video, or webcam index (0)")
    p.add_argument("--model", choices=["tiny", "full"], default="tiny")
    p.add_argument("--profile", default="pi4b", help="edge profile name or 'none'")
    p.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    p.add_argument("--nms", type=float, default=0.45, help="NMS IoU threshold")
    p.add_argument("--output", default=str(ROOT / "results"), help="output folder")
    p.add_argument("--show", action="store_true", help="display the video/webcam window")
    return p.parse_args()


def main():
    args = parse_args()
    if args.show and not gui_available():
        print("note: this OpenCV build has no window support; --show disabled (frames still saved)")
        args.show = False
    if args.profile == "none":
        profile = {"threads": 0, "input_size": 416, "description": "no throttle"}
    else:
        profile = edge_sim.get_profile(args.profile)
    print(edge_sim.describe(profile, args.profile))

    class_names = load_class_names(NAMES)
    net = build_net(args.model)
    out_names = output_layer_names(net)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    source = args.source
    path = Path(source)
    with edge_sim.apply_profile(profile):
        if source.isdigit() or path.suffix.lower() in VIDEO_EXT:
            run_stream(net, out_names, source, class_names, args, profile, out_dir)
        elif path.is_dir():
            images = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXT)
            run_images(net, out_names, images, class_names, args, profile, out_dir)
        elif path.suffix.lower() in IMAGE_EXT:
            run_images(net, out_names, [path], class_names, args, profile, out_dir)
        else:
            raise SystemExit(f"unsupported source: {source}")


if __name__ == "__main__":
    main()
