# Assignment 1: Identification of Leakages Using Hardware Performance Counters (HPCs) Profiling

## Overview
This project explores how **hardware performance counters (HPCs)** can be used to distinguish between different applications and analyze security vulnerabilities. Participants will use the **perf** tool to collect and analyze performance counter data across the following tasks:

## Tasks

### 1. Deep Neural Network (DNN) Fingerprinting
Inside the **DNN_Fingerprinting** directory, we have provided the script **"model_inference.py"**, which can be executed using:
```sh
python3 model_inference.py model_name
```
Possible model names:
- `alexnet`
- `resnet18`
- `vgg11`
- `densenet`
- `squeezenet`

#### Objective:
- Collect HPC traces for different architectures.
- Plot the collected data.
- Identify the **top 3 PMCs** (performance monitoring counters) that can effectively distinguish between these five architectures.

### 2. Class-Leakage Attack on PyTorch Models
Inside the **Class-Leakage** directory, we have provided the script **"CNN_infer.py"**, which can be executed using:
```sh
python3 CNN_infer.py --class_index <class_id> --image_index <image_id>
```
Example:
```sh
python3 CNN_infer.py --class_index 6 --image_index 10
```
This script will execute **10,000 inferences** of the specified class and image from the **CIFAR-10 dataset**.

#### Objective:
- Collect HPC data for **all 10 CIFAR-10 classes**.
- Identify the **top 3 HPCs** that can distinguish the **maximum class pairs** out of 45 (`10C2` combinations).
- Refer to the research paper for more insights: [Class-Leakage Attack](https://tches.iacr.org/index.php/TCHES/article/view/10295/9745)

## Objectives
- Learn to use **perf** for performance counter collection.
- Analyze **microarchitectural behavior** to fingerprint applications.
- Investigate **security risks** in different applications.
- Perform **statistical analysis** using **t-tests** to identify distinguishing HPCs.

## Prerequisites
- **Python 3**
- **perf tool** (Linux)
- **PyTorch** (for model execution)
- **Matplotlib** (for visualization)

## Installation
Ensure `perf` is installed:
```sh
sudo apt install linux-tools-common linux-tools-generic linux-tools-$(uname -r)
```

## Running Experiments

### 1. DNN Fingerprinting
Run inference on different deep learning models:
```sh
python3 model_inference.py model_name
```
Example:
```sh
python3 model_inference.py resnet18
```
Collect HPC traces, analyze them, and identify distinguishing PMCs.

### 2. Class-Leakage Attack
Run class-based inference and collect performance counters:
```sh
python3 CNN_infer.py --class_index <class_id> --image_index <image_id>
```
Example:
```sh
python3 CNN_infer.py --class_index 6 --image_index 10
```
Repeat for all classes and analyze distinguishing HPCs.

## Data Analysis
- Use **t-tests** to find HPCs that best distinguish architectures and classes.
- Plot results using **Matplotlib**.

## Submission Requirements
Participants must submit the following:
1. **Performance Counter Logs:** Raw logs collected using `perf`.
2. **Statistical Analysis:**
   - Conduct **t-test analysis** for each application.
   - Identify **three distinguishing HPCs** for both tasks.
3. **Justification:**
   - Explain why the selected HPCs effectively differentiate applications.
4. **Final Report:**
   - Summarize findings and observations in a structured format.

## References
- [Class-Leakage Attack Paper](https://tches.iacr.org/index.php/TCHES/article/view/10295/9745)

