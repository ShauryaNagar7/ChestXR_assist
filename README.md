# cxr-assist

Multi-label chest X-ray classification on CheXpert-small for three findings: Cardiomegaly, Pleural Effusion, Atelectasis.

**Status:** work in progress. EDA and baseline training are done; more experiments on the U-Ignore setup are planned.

## Problem

Chest X-rays are one of the most common imaging studies, and a single image can show several findings at once. This project treats it as a **multi-label** problem: for each X-ray, the model predicts an independent probability for each of three findings.

| Finding | What it is |
|---------|-----------|
| Cardiomegaly | Enlarged heart |
| Pleural Effusion | Fluid between the lung and chest wall |
| Atelectasis | Partial or complete lung collapse |

The long-term aim is a clinical decision-support tool built on top of the trained model. Right now the repo covers the model and its evaluation only.

## Dataset

[CheXpert-small on Kaggle](https://www.kaggle.com/datasets/ashery/chexpert), a downsampled version of the CheXpert dataset from Stanford (Irvin et al., 2019). The full dataset has about 224k chest radiographs from about 65k patients, labeled for 14 observations.

Things worth knowing about it:

- **Labels come from radiology reports**, not from manual image annotation. A rule-based NLP labeler extracted them from report text, so they carry some noise.
- **Multiple images per patient**, including frontal and lateral views. This project uses **frontal views only**, and splits must be done at the **patient level** to avoid leakage.
- **Image properties** (from EDA on a 5k frontal sample): all 8-bit grayscale, 0 unreadable files, height 320 with width 320-390, dataset mean 0.5063 / std 0.2895.

### Label values and uncertainty

Each observation is labeled one of:

| Value | Meaning |
|-------|---------|
| 1 | Positive: finding present |
| 0 | Negative: finding absent |
| -1 | **Uncertain**: the report hedges ("may represent...", "cannot exclude...") |
| blank | Not mentioned in the report (treated as 0 here) |

The uncertain (-1) labels are the main modelling decision. The finding may or may not be present, so how you treat them changes what the model learns. The common **uncertainty policies**:

- **U-Ignore:** mask uncertain labels out of the loss. No wrong labels injected, but less training signal.
- **U-Zeros:** treat uncertain as negative. Keeps all data, may under-call borderline findings.
- **U-Ones:** treat uncertain as positive. Keeps all data, may over-call.
- **U-MultiClass:** treat uncertain as a third class.

This project compares U-Ignore and U-Ones so far, with U-Zeros planned.

## Setup

- **Model:** DenseNet-121
- **Loss:** `BCEWithLogitsLoss`
- **Task:** multi-label classification (3 labels)
- **Compute:** Kaggle GPU
- **Uncertainty policies tried:** U-Ignore, U-Ones

## Done so far

- **EDA:** label distribution and uncertainty rates, image properties, frontal-view filtering
- **Training:** 3 epochs each with U-Ignore and U-Ones, best checkpoint picked by mean validation AUROC (both peaked at epoch 3)

## Results (validation, best epoch)

| Policy   | Cardiomegaly AUROC | Pleural Effusion AUROC | Atelectasis AUROC | Mean AUROC |
|----------|--------------------|------------------------|-------------------|------------|
| U-Ignore | 0.865              | 0.885                  | 0.703             | **0.8174** |
| U-Ones   | 0.864              | 0.885                  | 0.694             | 0.8141     |

AUPRC (U-Ignore): Cardiomegaly 0.548, Pleural Effusion 0.838, Atelectasis 0.321, mean 0.5691.
AUPRC (U-Ones): Cardiomegaly 0.542, Pleural Effusion 0.839, Atelectasis 0.309.

Each run takes about 70 min total (about 990s train + 420s eval per epoch, 3 epochs).

Both policies perform almost the same. Atelectasis is the weak label for both, which is consistent with it being a vaguer finding that reports tend to hedge on.

## Limitations

- Labels are NLP-extracted from reports, so metrics are bounded by label noise
- Trained on CheXpert-small (downsampled), not full-resolution images
- Short training (3 epochs); results are baselines, not tuned
- Single dataset, no external validation yet

## Next

- More work on U-Ignore
- U-Zeros
- Grad-CAM, calibration, subgroup analysis
- External validation on a second dataset