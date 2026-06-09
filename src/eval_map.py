"""Score detection accuracy on a slice of COCO val2017.

The first run pulls the validation annotations and however many images you ask
for, caches them under data/coco, and then runs the model over them. mAP comes
from pycocotools; precision, recall and F1 are taken at a single confidence
threshold by matching boxes to ground truth at IoU 0.5, which is the number
that actually tells you how the detector behaves at its working point.

Example:
    python src/eval_map.py --model tiny --num-images 200
"""

import argparse
import json
import urllib.request
import zipfile
from pathlib import Path

import cv2
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import edge_sim
from detect import build_net, infer
from utils import output_layer_names, rel

ROOT = Path(__file__).resolve().parent.parent
COCO_DIR = ROOT / "data/coco"
ANN_FILE = COCO_DIR / "annotations/instances_val2017.json"
IMG_DIR = COCO_DIR / "images"
ANN_ZIP_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
IMG_URL = "http://images.cocodataset.org/val2017/{:012d}.jpg"

# coco.names is ordered 0..79; COCO's own category ids skip a few numbers, so
# this maps a model class index onto the category id the annotations use.
COCO80_TO_91 = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
    85, 86, 87, 88, 89, 90,
]


def parse_args():
    p = argparse.ArgumentParser(description="COCO accuracy for YOLOv4-Tiny")
    p.add_argument("--model", choices=["tiny", "full"], default="tiny")
    p.add_argument("--num-images", type=int, default=200, help="how many val images to score")
    p.add_argument("--profile", default="pi4b", help="edge profile name or 'none'")
    p.add_argument("--conf", type=float, default=0.001, help="low threshold so the mAP curve is complete")
    p.add_argument("--op-conf", type=float, default=0.25, help="threshold for the precision/recall/F1 point")
    p.add_argument("--nms", type=float, default=0.45)
    return p.parse_args()


def ensure_annotations():
    """Grab instances_val2017.json out of the COCO annotations zip if we don't have it."""
    if ANN_FILE.exists():
        return
    COCO_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = COCO_DIR / "annotations_trainval2017.zip"
    print("fetching COCO val annotations (~241 MB, one time)...")
    urllib.request.urlretrieve(ANN_ZIP_URL, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        z.extract("annotations/instances_val2017.json", COCO_DIR)
    zip_path.unlink()


def ensure_images(image_ids):
    """Download any of the requested val images we haven't cached yet."""
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    missing = [i for i in image_ids if not (IMG_DIR / f"{i:012d}.jpg").exists()]
    for n, img_id in enumerate(missing, 1):
        urllib.request.urlretrieve(IMG_URL.format(img_id), IMG_DIR / f"{img_id:012d}.jpg")
        if n % 50 == 0:
            print(f"  downloaded {n}/{len(missing)} images")


def iou_xywh(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def operating_point(per_image, ground_truth, conf, iou_thr=0.5):
    """Walk every image, match detections to ground truth greedily by score, and
    tally true/false positives and misses to get precision, recall and F1."""
    tp = fp = fn = 0
    for img_id, gts in ground_truth.items():
        dets = sorted((d for d in per_image.get(img_id, []) if d["score"] >= conf),
                      key=lambda d: d["score"], reverse=True)
        used = [False] * len(gts)
        for d in dets:
            best_iou, best_j = iou_thr, -1
            for j, gt in enumerate(gts):
                if used[j] or gt["category_id"] != d["category_id"]:
                    continue
                value = iou_xywh(d["bbox"], gt["bbox"])
                if value >= best_iou:
                    best_iou, best_j = value, j
            if best_j >= 0:
                used[best_j] = True
                tp += 1
            else:
                fp += 1
        fn += used.count(False)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def main():
    args = parse_args()
    ensure_annotations()
    coco = COCO(str(ANN_FILE))
    image_ids = sorted(coco.getImgIds())[:args.num_images]
    ensure_images(image_ids)

    if args.profile == "none":
        profile = {"threads": 0, "input_size": 416}
    else:
        profile = edge_sim.get_profile(args.profile)
    input_size = profile.get("input_size", 416)

    net = build_net(args.model)
    out_names = output_layer_names(net)

    detections, per_image, ground_truth = [], {}, {}
    print(f"scoring {args.model} on {len(image_ids)} images at {input_size}x{input_size} ...")
    with edge_sim.apply_profile(profile):
        for k, img_id in enumerate(image_ids, 1):
            info = coco.loadImgs(img_id)[0]
            frame = cv2.imread(str(IMG_DIR / info["file_name"]))
            if frame is None:
                continue
            dets, _ = infer(net, out_names, frame, input_size, args.conf, args.nms)
            per_image[img_id] = []
            for class_id, score, (x, y, w, h) in dets:
                box = [float(x), float(y), float(w), float(h)]
                detections.append({"image_id": img_id, "category_id": COCO80_TO_91[class_id],
                                   "bbox": box, "score": float(score)})
                per_image[img_id].append({"category_id": COCO80_TO_91[class_id],
                                          "bbox": box, "score": float(score)})
            ground_truth[img_id] = [{"category_id": a["category_id"], "bbox": a["bbox"]}
                                    for a in coco.loadAnns(coco.getAnnIds(imgIds=img_id))]
            if k % 50 == 0:
                print(f"  {k}/{len(image_ids)}")

    coco_dt = coco.loadRes(detections)
    ev = COCOeval(coco, coco_dt, "bbox")
    ev.params.imgIds = image_ids
    ev.evaluate()
    ev.accumulate()
    ev.summarize()

    precision, recall, f1 = operating_point(per_image, ground_truth, args.op_conf)
    report = {
        "model": args.model,
        "input_size": input_size,
        "num_images": len(image_ids),
        "dataset": "COCO val2017 (first N image ids)",
        "map_50": round(float(ev.stats[1]), 4),
        "map_50_95": round(float(ev.stats[0]), 4),
        "operating_point": {
            "conf": args.op_conf,
            "iou": 0.5,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        },
    }
    out = ROOT / "results" / f"accuracy_{args.model}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print("\n" + json.dumps(report, indent=2))
    print(f"wrote {rel(out, ROOT)}")


if __name__ == "__main__":
    main()
