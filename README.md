# Real-Time Object Detection on Edge Devices with YOLOv4-Tiny

## 📌 Project Overview
This project studies **YOLOv4-Tiny** as a practical choice for real-time object detection on **edge devices** such as the Raspberry Pi. YOLOv4-Tiny is a stripped-down version of YOLOv4 that trades a little accuracy for a large gain in speed, which makes it a good fit for low-power, resource-constrained hardware. The work was done as a college task and is delivered as a presentation that walks through the architecture, the speed and accuracy trade-offs, and the optimizations that make the model feasible on the edge. The benchmark target throughout is a **Raspberry Pi 4B (4 GB)** running without a TPU.

---

## 🚀 Key Features
1. **Lightweight by design**: fewer convolution layers and a smaller backbone than YOLOv4, so it runs far faster on modest hardware.
2. **Real-time on the edge**: tuned for speed while keeping accuracy acceptable for everyday detection.
3. **Single-pass detection**: like the rest of the YOLO family, it frames detection as one regression pass over the image rather than a region-proposal pipeline.
4. **Edge-ready optimizations**: quantization, pruning, frame skipping, and optional TPU or GPU acceleration are all covered.

---

## 🧠 How YOLOv4-Tiny Works
The input image is resized to **416×416** and passed once through the network, which predicts bounding boxes, objectness scores, and class probabilities together. Detection runs at two scales, and Non-Max Suppression removes overlapping boxes at the end. The network has three parts:

1. **Backbone (CSPDarknet53-Tiny)**: efficient feature extraction using Cross Stage Partial blocks.
2. **Neck (Feature Pyramid Network)**: pools features across scales. Unlike YOLOv4 it drops SPP and PANet to stay light.
3. **Head (dual-scale)**: two feature maps, **13×13×255** for larger objects and **26×26×255** for smaller ones, each predicting box coordinates, a confidence score, and class probabilities.

Box regression uses **CIoU** loss, and training reuses the **Bag of Freebies** and **Bag of Specials** ideas from YOLOv4:
- **Bag of Freebies (BoF)**: accuracy gains that cost training time but not inference time (Mosaic augmentation, CutMix, Self-Adversarial Training, cosine learning-rate annealing).
- **Bag of Specials (BoS)**: small modules that add a little inference cost for an accuracy gain, such as Cross Stage Partial connections.

---

## 📊 Results
Benchmarked on a Raspberry Pi 4B (4 GB), without a TPU:

| Metric | YOLOv4 | YOLOv4-Tiny |
|---|---|---|
| Inference speed | ~2 FPS | **~14 FPS** |
| Precision | ~80% | ~70% |
| Recall | ~75% | ~68% |
| mAP@0.5 | ~55% | ~45% |
| F1-score | ~77% | ~71% |

On the Pi, YOLOv4-Tiny runs about **7× faster** than YOLOv4 for a modest accuracy drop, which is the trade-off that makes real-time edge detection workable.

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

## 📂 Repository Contents
- `Real Time Object Detection YOLOv4Tiny.pptx`: the full presentation, covering the YOLO timeline, the architecture in detail, the loss function, and the analysis summarized above.

---

## 🛠 System Requirements
- **Python**: 3.8+
- **Libraries**: `darknet`, `tensorflow-lite`, `opencv-python`, `numpy`
- **Hardware**: Raspberry Pi 4B (4 GB); a TPU or GPU is optional for extra speed.

---

## 📄 License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
