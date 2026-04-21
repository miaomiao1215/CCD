# CCD: Fine-Grained Image-Text Retrieval Benchmark with Controlled Contrastive Differences

This repository contains the official implementation and dataset construction pipeline for the paper:

**"Benchmarking the Fine-Grained Discriminability in Image-Text Retrieval via Controlled Contrastive Differences" (ACL 2026 Findings)**

---

## 🔍 Overview

Existing image-text retrieval benchmarks primarily evaluate **coarse-grained alignment**, while overlooking **fine-grained discriminability**.

To address this limitation, we introduce:

* **MSCOCO-CCD**
* **Flickr30k-CCD**

These datasets are built with **controlled contrastive differences (CCD)**, where each contrastive sample differs from the anchor in only **one fine-grained aspect**.

---

## 🧠 Key Contributions

* A **two-level image content taxonomy**:

  * Entity
  * Scene
  * Event/Action
  * Style & Presentation

* A **Human-LLM collaborative pipeline** for dataset construction

* A new evaluation metric:

  * **Fine-Grained Contrastive Discrimination Accuracy (FG-CDA)**

---

## 🏗️ Dataset Construction Pipeline

The dataset is constructed in three steps:

### Step 1: Contrastive Strategy Generation

* Extract key visual details using MLLMs
* Generate contrastive strategies based on taxonomy
* Filter infeasible or hallucinated strategies

### Step 2: Contrastive Image Generation

* Two methods:

  * Image search (Google Image API)
  * Image editing (e.g., Qwen-Image-Edit)
* Filter low-quality or inconsistent samples

### Step 3: Caption Refinement

* Generate fine-grained captions
* Ensure:

  * Accuracy
  * Discriminability
  * Inclusion of contrastive details

---

## 📂 Project Structure

```
.
├── dataset_construction/
│   ├── step1_contrastive_strategy.py
│   ├── step2_edit_image.py
│   ├── step3_generate_fine_captions.py
│   ├── prompt.py
├── evaluation/
│   ├── evaluate.py
│   └── metrics.py
├── assets/
│   ├── example.jpg
│   └── pipeline.jpg
```

---

## 🚀 Usage

### 1. Dataset Construction

```bash
python dataset_construction/step1_contrastive_strategy.py
python dataset_construction/step2_edit_image.py
python dataset_construction/step3_generate_fine_captions.py
```

---

### 2. Evaluation

```bash
python evaluation/evaluate.py
```

---

## 📊 Evaluation Metric

We propose **FG-CDA (Fine-Grained Contrastive Discrimination Accuracy)**:

A retrieval is correct if:

> The similarity between query and target > similarity between query and contrastive sample

This measures the ability to distinguish **subtle visual differences**.

---

## 📥 Dataset

Due to size limitations, we provide:

* 🔗 Full dataset: [https://drive.google.com/drive/folders/1VsrclyEMOXqkHlGoO6ratcI6PeYOvnIG]

---

## 📜 License

All images are collected under licenses that allow redistribution:

* Public Domain
* Creative Commons (CC-BY, CC-SA, CC-NC)

---

## ⭐ Star this repo if you find it useful!
