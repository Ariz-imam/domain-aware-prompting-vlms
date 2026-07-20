# Domain-Aware Prompting for Zero-Shot Vision-Language Models under RGB–Thermal Sensor Domain Shift

## Overview

This repository contains a small empirical study investigating whether **simple domain-aware prompts** can improve the zero-shot performance of CLIP under a real RGB–thermal sensor domain shift.

This is **not** a new prompting method, a new model, or a paper draft. The goal is simply to evaluate a focused research question through a controlled experiment.

The motivation comes from Prof. Niloufar (Nila) Alipour Talemi's **Style-Pro** work, which demonstrates that style-aware prompts can improve the generalization of vision-language models across artistic domains such as sketches, paintings, and cartoons.

Rather than reproducing those experiments, this project asks whether the same intuition transfers to a different and practically important setting:

> **Can simple domain-aware prompts improve zero-shot CLIP performance under RGB–thermal sensor domain shift without modifying the model?**

---

# Why this experiment?

Most prior work on prompt learning evaluates domain shifts such as:

- Photo → Sketch
- Photo → Painting
- Photo → Cartoon

These settings are already well studied.

Instead, this project focuses on a **sensor domain shift**, where the visual appearance changes because images are captured using different sensing modalities (RGB versus thermal imaging).

Sensor-domain shifts naturally appear in applications such as:

- autonomous driving
- night-time perception
- remote sensing
- wildfire monitoring

To the best of my literature review, I found limited evaluations of **simple handcrafted domain-aware prompting** for RGB–thermal zero-shot classification using CLIP.

---

# Dataset

**Teledyne FLIR ADAS Thermal Dataset v2**

Source:
`samdazel/teledyne-flir-adas-thermal-dataset-v2` (Kaggle)

Dataset characteristics:

- paired RGB and thermal images
- vehicle-mounted camera platform
- COCO-format bounding-box annotations
- real outdoor driving scenes

Classes evaluated:

- person
- car
- bike
- truck

These classes were selected because they contain sufficient samples in both RGB and thermal domains.

For each class:

- 150 RGB crops
- 150 thermal crops

Total:

**1,200 object crops**

Bounding boxes were cropped with a small padding margin before inference.

---

# Method

Model:

- CLIP ViT-B/32

Training:

- none

Fine-tuning:

- none

Everything is performed in the **zero-shot setting**.

Four prompt conditions are compared on the **same image crops**, ensuring a controlled comparison.

| Condition | Prompt |
|-----------|--------|
| **Bare** | `car` |
| **Generic** | `a photo of a {class}` |
| **Domain-Aware** | `a thermal camera image of a {class}` (thermal) / `a color photograph of a {class}` (RGB) |
| **Domain-Aware Ensemble** | Multiple domain-aware prompt variants whose text embeddings are averaged |

RGB acts as a **control condition**.

Because the generic and domain-aware prompts are intentionally very similar for RGB images, improvements observed primarily on thermal imagery are more likely to reflect adaptation to the sensing modality rather than prompt wording alone.

---

# Results

## Thermal (primary experiment)

| Prompt | Accuracy |
|---------|---------:|
| Bare | **44.5%** |
| Generic | **44.0%** |
| Domain-Aware | **50.7%** |
| Domain-Aware Ensemble | **51.7%** |

The domain-aware prompt improves zero-shot thermal accuracy by **6.7 percentage points** over the generic prompt.

The prompt ensemble provides a small additional improvement.

---

## RGB (control experiment)

| Prompt | Accuracy |
|---------|---------:|
| Generic | **69.0%** |
| Domain-Aware | **70.7%** |

Only a very small improvement is observed on RGB images.

This supports the idea that the thermal improvement is associated with the sensor-domain shift rather than prompt wording in general.

---

# Class-wise observations

The improvements are **not uniform across classes**.

- Truck shows the largest improvement.
- Car remains largely unchanged.
- Person recall decreases slightly as the model becomes less biased toward predicting "person."
- Bike remains difficult under all prompt conditions.

This suggests that domain-aware prompting helps some semantic categories more than others.

---

# Failure Analysis

Representative failure cases indicate that most incorrect predictions occur when:

- objects occupy only a very small region of the image,
- thermal contrast is weak,
- object boundaries are poorly defined.

These examples illustrate the remaining limitations of zero-shot CLIP even after domain-aware prompting.

---

# Discussion

This experiment demonstrates that **simple prompt modifications alone** can improve zero-shot performance under a genuine sensor-domain shift without retraining or fine-tuning the model.

The improvements are moderate rather than dramatic, and they are concentrated in particular classes rather than being uniform across the dataset.

While this is a small empirical study rather than a new prompting method, the results suggest that prompt design can serve as a lightweight adaptation strategy when vision-language models are applied to new sensing modalities.

---

# Limitations

- Only one dataset (FLIR ADAS) was evaluated.
- Prompt templates were manually designed.
- Only one vision-language model (CLIP ViT-B/32) was tested.
- Improvements were class-dependent rather than uniform.
- No prompt optimization or prompt learning was performed.

---

# Repository Structure

```
.
├── style_aware_prompting_flir.ipynb
├── README.md
└── results
    ├── figures
    ├── metrics
    └── run_manifest.json
```

---

# Reproducibility

1. Create a Kaggle notebook.
2. Attach the **Teledyne FLIR ADAS Thermal Dataset v2**.
3. Enable Internet access (required once to download CLIP weights).
4. Enable a GPU (T4 is sufficient).
5. Run the notebook from top to bottom.

All outputs are automatically saved under:

```
results/
```

including:

- figures
- raw predictions
- summary metrics
- experiment configuration

---

# Acknowledgement

This project was inspired by the ideas presented in **Style-Pro** by Prof. Niloufar (Nila) Alipour Talemi and colleagues. It does **not** reproduce or extend Style-Pro directly; instead, it investigates whether the intuition behind domain-aware prompting transfers to a practical RGB–thermal sensor-domain shift using a simple zero-shot evaluation.
