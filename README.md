# Real-Time Object Detection on Edge Devices with YOLOv4-Tiny

## Project Overview
This project demonstrates real-time object detection using YOLOv4-Tiny on edge devices like the Raspberry Pi. The lightweight architecture enables high-speed inference and is optimized for resource-constrained hardware.

## Key Features
- **Lightweight Architecture**: YOLOv4-Tiny uses fewer layers and smaller backbones, making it faster and suitable for edge deployment.
- **Real-Time Performance**: Prioritizes speed and efficiency with acceptable accuracy for real-world applications.
- **Edge Device Optimization**: Tested on devices like Raspberry Pi 4B with methods like quantization, pruning, and TPU acceleration.

## Importance of Real-Time Object Detection
- **Applications**: Surveillance, autonomous vehicles, healthcare, robotics.
- **Time-Sensitive Decisions**: Enables instantaneous responses in dynamic environments.

## YOLOv4-Tiny Architecture
1. **Backbone**: CSPDarknet53-Tiny for feature extraction.
2. **Neck**: Feature Pyramid Network (FPN) for feature pooling.
3. **Head**: Dual-scale detection for large and small objects using 13×13 and 26×26 grids.

## Performance Enhancements
- **Bag of Freebies (BoF)**: Techniques like CutMix, Mosaic Augmentation, and SAT for better training without impacting inference time.
- **Bag of Specials (BoS)**: Techniques to improve accuracy with minimal inference time impact (e.g., CSP connections).
- **Quantization and Pruning**: Reduces model size and increases speed.

## Advantages of YOLOv4-Tiny
- **Fewer Parameters**: ~6 million compared to ~63.6 million in YOLOv4.
- **Simpler Activation**: LeakyReLU over Mish activation.
- **Efficient Neck Design**: Uses FPN instead of SPP and PANet.

## Evaluation and Metrics
- Model performance compared to YOLOv4 on Raspberry Pi 4B:
  - **Metrics Evaluated**: FPS, inference time, accuracy.
  - **Hardware Configurations**: Tested without and with TPU/GPU acceleration.

## Tweaking and Optimization
- Use of TPU (e.g., Google Coral).
- Frame skipping for performance boost.
- Quantization and model pruning for improved speed and reduced size.

## Conclusion
YOLOv4-Tiny is a practical solution for deploying real-time object detection on resource-constrained edge devices. It balances speed and accuracy, making it feasible for time-critical applications.
