# Smart Home Face Recognition & Anti-Spoofing Door Lock System

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Raspberry Pi](https://img.shields.io/badge/Raspberry%20Pi-4B%20(2GB)-C51A4A?style=for-the-badge&logo=Raspberry-Pi&logoColor=white)](https://www.raspberrypi.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![dlib](https://img.shields.io/badge/dlib-ResNet--128D-00599C?style=for-the-badge)](http://dlib.net/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

An intelligent, real-time biometric access control system engineered for smart homes using Raspberry Pi 4. The system combines computer vision, deep learning embeddings, an Eye Aspect Ratio (EAR) anti-spoofing state machine, and hardware-level PWM motor control to deliver touchless, highly secure, and jitter-free door actuation.

> **Project Report**: University of Information Technology - VNU-HCM (UIT), Faculty of Computer Engineering.  
> **Authors**: Tran Quang Nhat (23521102) & Ngo Tien Dat (23520254)  
> **Advisor**: M.Sc. Phan Dinh Duy  

---

## 📋 Table of Contents
- [Overview & Key Features](#-overview--key-features)
- [System Architecture & Flowchart](#-system-architecture--flowchart)
- [Hardware Architecture & Power Design](#-hardware-architecture--power-design)
- [Key Algorithms & Mathematics](#-key-algorithms--mathematics)
  - [1. Face Detection & Feature Embedding](#1-face-detection--feature-embedding)
  - [2. EAR Anti-Spoofing State Machine](#2-ear-anti-spoofing-state-machine)
  - [3. Optimization Techniques](#3-optimization-techniques)
- [Hardware Components List](#-hardware-components-list)
- [Installation & Setup](#-installation--setup)
- [Project Directory Structure](#-project-directory-structure)

---

## 🌟 Overview & Key Features

Traditional mechanical keys, numeric PIN pads, and RFID cards suffer from loss, copying, physical tampering, and shoulder surfing. Standard 2D face recognition systems are vulnerable to printed photo attacks. 

This project solves these limitations by deploying a low-latency, highly resilient biometric door lock pipeline on budget hardware (Raspberry Pi 4 - 2GB RAM):

- 🔒 **High Security (`TOLERANCE = 0.45`)**: Strict Euclidean distance thresholding eliminates false positives (FAR < 1%).
- 🛡️ **Liveness Verification (EAR State Machine)**: Prevents 2D photo/display spoofing by enforcing a biological eye-closure challenge (*Open eyes → Hold closed eyes for 2.0s → Unlock*).
- ⚡ **Real-Time Edge Optimization**: Achieves 20 FPS preview and 6–10 FPS recognition using AI frame downscaling (240px width) and dynamic frame skipping (`dynamic_skip = 4` idle / `0` active).
- 👁️ **Anti-Glare / Backlight Compensation**: Integrates CLAHE (Contrast Limited Adaptive Histogram Equalization) in the YCrCb color space to maintain high detection accuracy in backfilled or high-contrast illumination.
- ⚙️ **Jitter-Free Motor Actuation**: Driven by `pigpio` DMA hardware PWM and automatic detach (`servo.detach()`), completely eliminating motor buzzing, overheating, and mechanical gear wear.
- 🔌 **Isolated Power Management**: Dual-rail power separation prevents Raspberry Pi brownouts caused by high servo stall currents.
- ⏱️ **Auto-Lock Safety**: Automatically re-locks the door mechanism 5 seconds after access is granted.

---

## 📐 System Architecture & Flowchart

```text
                          +------------------------+
                          |   Raspberry Pi Camera  |
                          |     (OV5647 5MP)       |
                          +-----------+------------+
                                      |
                                      v
                          +------------------------+
                          |  Picamera2 Stream Capture |
                          |    (640x480 BGR/RGB)   |
                          +-----------+------------+
                                      |
                                      v
                          +------------------------+
                          | Frame Resize (240px)   |
                          |  + CLAHE Contrast Fix  |
                          +-----------+------------+
                                      |
                                      v
                          +------------------------+
                          | Face Detection (HOG)   |
                          +-----------+------------+
                                      |
                            [Face Detected?]
                             /            \
                           Yes             No ----> Reset State / Resume
                            |
                            v
                          +------------------------+
                          | ResNet 128D Embedding  |
                          |  + Euclidean Distance  |
                          +-----------+------------+
                                      |
                            [Distance < 0.45?]
                             /            \
                           Yes             No ----> Label "Unknown" (Red Box)
                            |
                            v
                    +-------------------------------+
                    | EAR Anti-Spoofing State Machine|
                    +---------------+---------------+
                                    |
                    +---------------+---------------+
                    | State 0: Eye Open Check (EAR>0.3)|
                    +---------------+---------------+
                                    |
                    +---------------+---------------+
                    | State 1: Hold Closed (EAR<0.22)|
                    |         for >= 2.0s          |
                    +---------------+---------------+
                                    |
                            [Timer >= 2.0s?]
                             /            \
                           Yes             No ----> Display Prompt & Hold
                            |
                            v
                    +-------------------------------+
                    | State 2: UNLOCK DOOR          |
                    | (Hardware PWM Servo Trigger)  |
                    +---------------+---------------+
                                    |
                    +---------------+---------------+
                    | 5s Countdown -> AUTO-LOCK     |
                    +-------------------------------+
```

---

## ⚡ Hardware Architecture & Power Design

Servo motors like the MG996R draw instantaneous stall currents up to 1.5A–2.0A when engaging the locking mechanism. Connecting high-torque servos directly to the Raspberry Pi 5V power bus causes voltage sags below 4.75V, triggering brownout resets (`Low Voltage Warning`).

### Power Isolation Schema
- **Logic Rail**: Raspberry Pi 4 is powered by a dedicated 5V/3A power adapter.
- **Actuator Rail**: MG996R Servo is powered by 2x 18650 Li-ion batteries (7.4V nominal) stepped down to 5.0V/3A via an LM2596 DC-DC Buck Converter.
- **Common Ground**: Battery negative terminal is tied directly to the Raspberry Pi `GND` pin to establish a common voltage reference for PWM signaling.

```text
       [ 2x 18650 Li-ion (7.4V) ]
                   |
                   v
       [ LM2596 Buck Converter (5V/3A) ]
            |                      |
         (+5V Rail)             (GND Rail)
            |                      |
            +-------> [ Servo VCC ]| 
                      [ Servo GND ]+---------------------+
                                                           |
       [ Raspberry Pi 5V Power ]                          |
            |                                              |
       [ Raspberry Pi 4B ]                                 |
            |-- GPIO 17 (PWM Control) ---> [ Servo Signal ]|
            |-- GND ---------------------------------------+ (Common GND)
```

---

## 🔬 Key Algorithms & Mathematics

### 1. Face Detection & Feature Embedding

- **HOG (Histogram of Oriented Gradients)**: Computes local light intensity gradients across $8 \times 8$ pixel cells and $16 \times 16$ blocks. Normalized feature vectors are passed through a Linear SVM to locate face bounding boxes efficiently on ARM CPUs without GPU reliance.
- **ResNet 128D Embeddings**: Crop regions are transformed into a 128-dimensional vector space using dlib's Deep Residual Network trained with Triplet Loss:

   $$\mathcal{L}_{triplet} = \max(0, \|f(A) - f(P)\|^2 - \|f(A) - f(N)\|^2 + \alpha)$$
  
- **Euclidean Metric Matching**:
  
  $$\text{Distance} = \sqrt{\sum_{i=1}^{128} (V_{\text{realtime}}[i] - V_{\text{database}}[i])^2}$$
  
  - Default threshold: `0.60` (lenient).
  - Production threshold: `0.45` (strict security enforcement).

### 2. EAR Anti-Spoofing State Machine

Liveness is validated using 6 eye landmark points $(P_0 \dots P_5)$ to calculate the Eye Aspect Ratio (EAR):

$$\text{EAR} = \frac{\|P_1 - P_5\| + \|P_2 - P_4\|}{2 \times \|P_0 - P_3\|}$$

```text
    P1  P2
  P0 +----+ P3        EAR > 0.30 : Eye Open
    P5  P4            EAR < 0.22 : Eye Closed
```

#### State Machine Transition:
1. **State 0 (Open Eye Requirement)**: System verifies user is present with open eyes ($	ext{EAR} > 0.30$).
2. **State 1 (Closed Eye Challenge)**: System checks if user closes eyes ($	ext{EAR} < 0.22$). Timer initializes (`eyes_closed_start_time`). Must maintain closure for **$\ge 2.0$ seconds**.
3. **State 2 (Unlock Trigger)**: `doorUnlock = True`, Servo rotates to 90° (`servo.mid()`), timer resets, and 5-second auto-lock countdown begins.

### 3. Optimization Techniques

- **Resolution Downscaling**: Image width is downscaled to 240px for AI execution, saving >60% CPU cycles, then bounding box coordinates are scaled back ($	imes \text{scale}$) for display rendering (640x480).
- **Dynamic Frame Skipping**: Skip factor is set to `dynamic_skip = 4` while scanning. Once a valid face is recognized and waiting for liveness verification, `dynamic_skip` immediately drops to `0` to prevent missing eye blinks.
- **CLAHE (Backlight Filter)**: Frame is converted to `YCrCb`, CLAHE (`clipLimit=2.0`, `tileGridSize=(8,8)`) is applied to the Y (luminance) channel, and re-merged to BGR.

---

## 🛠️ Hardware Components List

| Component | Specification / Description | Quantity |
| :--- | :--- | :---: |
| **Raspberry Pi 4 Model B** | Broadcom BCM2711, Quad-core Cortex-A72 @ 1.5GHz, 2GB LPDDR4 RAM | 1 |
| **Camera Module** | Raspberry Pi Camera OV5647 5MP (160° Wide Angle, CSI Interface) | 1 |
| **Servo Motor** | MG996R Metal Gear High-Torque Servo (9.4 kg/cm @ 4.8V) | 1 |
| **Buck Converter** | LM2596 DC-DC Step-Down Module (Input 7-40V, Output 5V 3A) | 1 |
| **Power Supply** | 2x 18650 Li-ion Batteries (7.4V Total) + 18650 Holder | 1 |
| **Storage** | 32GB MicroSD Card (Class 10 U3) | 1 |
| **Chassis / Frame** | Custom Formex door prototype with mechanical hinge & latch | 1 |

---
## ⚙️ Installation & Setup

### 1. Prerequisites & System Packages
Update system packages and install underlying C++ dependencies for OpenCV and dlib on Raspberry Pi OS (Debian/Raspbian):

```bash
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y build-essential cmake pkg-config \
    libjpeg-dev libtiff5-dev libpng-dev \
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
    libxvidcore-dev libx264-dev libfontconfig1-dev libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev libopenblas-dev liblapack-dev gfortran \
    python3-dev python3-pip pigpio
```

### 2. Enable Pigpio Daemon (Hardware PWM)
`pigpio` provides microsecond-precise DMA hardware PWM to drive the servo motor without CPU timing jitter.

```bash
sudo systemctl enable pigpiod
sudo systemctl start pigpiod
```

### 3. Python Environment Setup
Install required Python packages:

```bash
pip3 install opencv-python numpy imutils face_recognition picamera2 gpiozero
```

---
## 📁 Project Directory Structure

```text
Smarthome_FaceRecognition/
├── dataset/                    # Raw face image repository
│   ├── User_1/                 # 10 sampled images for User 1
│   ├── User_2/                 # 10 sampled images for User 2
│   └── ...                     # Additional enrolled users
├── encodings.pickle            # Serialized 128D biometric vector database
├── face_shot.py                # Module 1: Camera initialization & dataset sampling
├── train_model.py              # Module 2: Feature extraction & pickle serialization
├── face_recog.py               # Module 3: Real-time recognition, anti-spoofing & servo control
└── README.md                   # System documentation
```









