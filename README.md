# Real-Time Object Detection on Edge Devices with YOLOv4-Tiny

## 📌 Project Overview
This project studies **YOLOv4-Tiny** as a practical choice for real-time object detection on **edge devices** such as the Raspberry Pi. YOLOv4-Tiny is a stripped-down version of YOLOv4 that trades a little accuracy for a large gain in speed, which makes it a good fit for low-power, resource-constrained hardware. It started as a college task delivered as a presentation on the architecture, the speed and accuracy trade-offs, and the optimizations that make the model feasible on the edge.

The repository now also carries a small runnable demo around that deck. It loads pretrained COCO weights through OpenCV on the CPU, runs detection on images and video, and scores accuracy on a slice of COCO val. It does not need a physical Raspberry Pi: instead it **simulates the edge envelope on a constrained CPU** (CPU-only, a fixed thread count, a set input size), which keeps the speed and accuracy trade-offs real and reproducible on any laptop. The device the cited figures are framed against is a **Raspberry Pi 4B (4 GB)** without a TPU.

---

## 🚀 Key Features
1. **Lightweight by design**: fewer convolution layers and a smaller backbone than YOLOv4, so it runs far faster on modest hardware.
2. **Real-time on the edge**: tuned for speed while keeping accuracy acceptable for everyday detection.
3. **Single-pass detection**: like the rest of the YOLO family, it frames detection as one regression pass over the image rather than a region-proposal pipeline.
4. **Runnable and measured**: CPU-only inference, an edge-profile benchmark, and a COCO accuracy script, so the numbers in this README are produced by the code in it.

---

## 🧠 How YOLOv4-Tiny Works
The input image is resized to **416×416** and passed once through the network, which predicts bounding boxes, objectness scores, and class probabilities together. Detection runs at two scales, and Non-Max Suppression removes overlapping boxes at the end. The network has three parts:

1. **Backbone (CSPDarknet53-Tiny)**: efficient feature extraction using Cross Stage Partial blocks.
2. **Neck (Feature Pyramid Network)**: pools features across scales. Unlike YOLOv4 it drops SPP and PANet to stay light.
3. **Head (dual-scale)**: two feature maps, **13×13×255** for larger objects and **26×26×255** for smaller ones, each predicting box coordinates, a confidence score, and class probabilities.

![YOLOv4-Tiny architecture](docs/architecture.png)

*CSPDarknet53-Tiny backbone feeding an FPN neck and two YOLO heads at 13×13 and 26×26 (from the project presentation).*

Box regression uses **CIoU** loss, and training reuses the **Bag of Freebies** and **Bag of Specials** ideas from YOLOv4:
- **Bag of Freebies (BoF)**: accuracy gains that cost training time but not inference time (Mosaic augmentation, CutMix, Self-Adversarial Training, cosine learning-rate annealing).
- **Bag of Specials (BoS)**: small modules that add a little inference cost for an accuracy gain, such as Cross Stage Partial connections.

---

## 📁 Repository Layout
```
yolov4-tiny-edge/
├── README.md
├── LICENSE
├── Real Time Object Detection YOLOv4Tiny.pptx   # the original deck
├── Dockerfile                                   # Pi-4B-like container
├── requirements.txt                             # core demo deps
├── requirements-eval.txt                        # extra deps for accuracy eval
├── docs/
│   └── architecture.png
├── configs/
│   └── edge_profiles.yaml                        # named device envelopes
├── models/
│   └── download_weights.py                       # fetches cfg/weights/names
├── data/
│   └── samples/                                  # public test images
├── src/
│   ├── detect.py                                 # image/video/webcam inference
│   ├── benchmark.py                              # FPS and latency, tiny vs full
│   ├── eval_map.py                               # COCO mAP / precision / recall
│   ├── edge_sim.py                               # edge-profile throttle
│   └── utils.py                                  # letterbox, decode, draw, timing
└── results/                                      # committed benchmark + detection artefacts
```

---

## 🏃 Run it
The demo runs entirely on the CPU through OpenCV's DNN module, so it needs no darknet build and no GPU. It pulls the headless OpenCV wheel, which installs cleanly on Linux, macOS and Windows (and inside Docker) with no system graphics libraries.

```bash
# 1. create and activate a virtual environment
python3 -m venv .venv                 # Windows: py -m venv .venv
source .venv/bin/activate             # Windows: .venv\Scripts\activate

# 2. install dependencies
pip install -r requirements.txt

# 3. download pretrained COCO weights (tiny cfg + weights + class names, ~24 MB)
python models/download_weights.py
#   add --full to also fetch YOLOv4 (~250 MB) for the speed comparison

# 4. detect on the sample images
python src/detect.py --source data/samples --profile pi4b

# 5. benchmark FPS and latency across edge profiles
python src/benchmark.py --models tiny --profiles unconstrained,pi4b,pi4b_320,edge_2core
```

After `activate`, `python` and `pip` point at the virtualenv, so steps 3 to 5 are identical on Linux, macOS and Windows.

`detect.py` takes an image, a folder, a video file, or a webcam index (`--source 0`), writes annotated frames to `results/`, and prints per-frame latency and FPS. Useful flags: `--model tiny|full`, `--profile <name>|none`, `--conf`, `--nms`, and `--show` for a live window (only on a GUI OpenCV build).

**Edge profiles** live in `configs/edge_profiles.yaml`. Each one fixes a thread count and input size to imitate a device:

| Profile | Threads | Input | Imitates |
|---|---|---|---|
| `unconstrained` | all cores | 416 | dev CPU baseline |
| `pi4b` | 4 | 416 | Raspberry Pi 4B (4 GB) |
| `pi4b_320` | 4 | 320 | Raspberry Pi 4B, smaller input |
| `edge_2core` | 2 | 256 | tighter 2-core envelope |

The thread part of a profile is enforced from Python and works on any OS. To hold the *whole* machine to a device envelope you constrain it from the outside: `taskset` pins physical cores on Linux, and the Docker route caps cores and memory on macOS and Windows as well.

```bash
# Linux: pin to 4 cores
taskset -c 0-3 python src/benchmark.py --profiles pi4b

# any OS with Docker: cap cores and memory to a Raspberry Pi 4B budget
docker build -t yolov4-tiny-edge .
docker run --rm --cpus=4 --memory=4g -v "$PWD/results:/app/results" yolov4-tiny-edge
#   PowerShell uses ${PWD}; cmd uses %cd%
```

To measure accuracy on COCO val (this needs the extra eval dependency):

```bash
pip install -r requirements-eval.txt
python src/eval_map.py --model tiny --num-images 200
```

---

## 📊 Results

### Cited figures (Raspberry Pi 4B 4GB, no TPU)
These come from the original presentation and typical Pi 4B benchmarks in the literature. They are quoted for context and were not re-measured in this repo.

| Metric | YOLOv4 | YOLOv4-Tiny |
|---|---|---|
| Inference speed | ~2 FPS | **~14 FPS** |
| Precision | ~80% | ~70% |
| Recall | ~75% | ~68% |
| mAP@0.5 | ~55% | ~45% |
| F1-score | ~77% | ~71% |

On the Pi, YOLOv4-Tiny runs about **7× faster** than YOLOv4 for a modest accuracy drop, which is the trade-off that makes real-time edge detection workable.

### Measured speed on an edge-simulated CPU profile
These numbers come from `src/benchmark.py` on my dev machine (AMD Ryzen 7 7840HS, 16 cores, WSL2), CPU-only, restricted to each profile's thread count and input size. They are not from a Raspberry Pi. A laptop core is much faster than a Pi's Cortex-A72, so the absolute FPS sits well above the ~14 FPS cited above. What carries over is the relative trade-off and a reproducible method. Absolute numbers also shift a little run to run with background load.

YOLOv4-Tiny, single 768×576 image, 5 warmup and 30 timed iterations ([results/benchmark.json](results/benchmark.json)):

| Profile | Threads | Input | Latency (mean) | p95 | FPS |
|---|---|---|---|---|---|
| unconstrained | 16 | 416 | 25.3 ms | 29.2 ms | 39.6 |
| pi4b | 4 | 416 | 22.4 ms | 23.5 ms | 44.6 |
| pi4b_320 | 4 | 320 | 15.4 ms | 17.6 ms | 64.9 |
| edge_2core | 2 | 256 | 15.9 ms | 16.7 ms | 62.8 |

![FPS by edge profile](results/fps_plot.png)

Two things stand out. Thread count barely matters for a model this small: at 416 input, 4 threads and all 16 land within a few FPS of each other (44.6 vs 39.6 here), so tiny saturates the CPU well before 16 cores. Input size is what moves the needle: going from 416 to 320 at the same 4 threads lifts FPS from 44.6 to 64.9.

### Tiny vs full YOLOv4 (measured)
Full YOLOv4 at 416 needs more memory than this box had free and gets OOM-killed, so it was benchmarked at the input sizes that fit (320 and 256). Tiny runs at every profile, and that memory headroom is part of why tiny is the edge choice. The tiny figures here are the same measurement as the speed table above ([results/benchmark.json](results/benchmark.json)); full's come from [results/benchmark_full.json](results/benchmark_full.json), same image and iteration counts.

| Profile | Input | YOLOv4-Tiny FPS | YOLOv4 FPS | Tiny speedup |
|---|---|---|---|---|
| pi4b_320 | 320 | 64.9 | 6.2 | 10.5× |
| edge_2core | 256 | 62.8 | 5.5 | 11.4× |

![Tiny vs full FPS](results/fps_plot_compare.png)

On this CPU tiny runs **10 to 11× faster** than full, more than the cited ~7× on the Pi.

### Measured accuracy on a COCO val subset
`src/eval_map.py` scores the model on a slice of COCO val2017. It downloads the annotations and images on first use (cached under `data/coco`, gitignored), runs the detector, and reports mAP from pycocotools plus a precision/recall/F1 point from IoU matching.

YOLOv4-Tiny on the first 200 val2017 images at 416 input ([results/accuracy_tiny.json](results/accuracy_tiny.json)):

| Metric | Value |
|---|---|
| mAP@0.5 | 33.8% |
| mAP@0.5:0.95 | 20.8% |
| Precision @ conf 0.25 | 83.9% |
| Recall @ conf 0.25 | 27.9% |
| F1 @ conf 0.25 | 41.9% |

mAP is the threshold-free number to compare against, and 33.8% at IoU 0.5 is in line with the ~40% tiny reaches on the full 5k-image val set; the gap is mostly the small 200-image sample. The precision and recall are read at the detector's default 0.25 confidence, where it leans cautious: it is right about 84% of the boxes it draws but only finds 28% of every labelled object, and COCO is full of the small, crowded objects tiny tends to miss. Lowering the threshold trades precision back for recall. These figures are measured here and differ from the cited Pi table above, which used a different benchmark and threshold.

---

## 🖼 Detection Examples
The same image through both models. Full YOLOv4 reads the vehicle as a truck and scores the dog and bicycle higher; tiny is faster but coarser and calls the truck a car.

| YOLOv4-Tiny | full YOLOv4 |
|---|---|
| ![tiny](results/dog_tiny.jpg) | ![full](results/dog_full.jpg) |

More YOLOv4-Tiny detections on the sample set:

| | |
|---|---|
| ![person](results/person_tiny.jpg) | ![horses](results/horses_tiny.jpg) |

---

## ⚡ What Makes It Fast
- Fewer convolution layers than YOLOv4.
- Simpler CSP blocks.
- LeakyReLU activation instead of Mish.
- Around 6M parameters versus 63.6M in YOLOv4.
- An FPN-only neck (no SPP, no PANet).
- Two detection heads instead of three.

---

## 🔧 Going Further
The presentation also covers ways to push speed past the vanilla model:
- A TPU (Google Coral) or a GPU for hardware acceleration.
- Frame skipping on the video stream.
- Post-training quantization.
- Model pruning.

---

## 🛠 System Requirements
- **Python**: 3.8+
- **Platforms**: Linux, macOS (Intel and Apple Silicon), and Windows. Every dependency ships a prebuilt wheel, so `pip install` pulls no system packages. The only Linux-only command is `taskset`; the rest, including the Docker envelope, runs the same everywhere.
- **Core libraries**: `opencv-python-headless`, `numpy`, `matplotlib`, `pyyaml` (see `requirements.txt`).
- **Accuracy eval**: adds `pycocotools` (see `requirements-eval.txt`).
- **Compute**: runs CPU-only, no GPU or darknet build needed. The model loads through OpenCV's DNN module. Optional: Docker for the throttled container, a TPU or GPU on real hardware for extra speed.
- **Target device**: Raspberry Pi 4B (4 GB) for the cited figures.

---

## 📄 License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
