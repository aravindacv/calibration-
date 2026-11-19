# Label-free shift detection and few-label recalibration for skin lesion classifiers

Code and derived results for the paper

> Aravinda C. V., A. D. Shetty, U. K. Reddy K. R., R. Bhatnagar. *Label-free shift detection and few-label recalibration for trustworthy skin lesion classifiers.* Submitted to *Computerized Medical Imaging and Graphics*.

The study trains 24 EfficientNet-B0/B3 classifiers on MILK10k dermoscopy (cross-entropy and focal loss, six seeds each) and evaluates calibration under a trunk-site shift, HAM10000 dermoscopy and PAD-UFES-20 smartphone photographs. It proposes a label-free shift alarm (Wasserstein matching of confidence distributions against corrupted source data), few-label temperature recalibration, and a centre-controlled Grad-CAM faithfulness test.

## Contents

| Path | What it is |
|---|---|
| `calibration_shift_milk10k.ipynb` | All experiment cells, in the order they were run (Kaggle, GPU T4 for training and Grad-CAM). |
| `results/` | Derived files that reproduce every table and figure without retraining (see below). |

## Notebook sections and paper mapping

| Notebook section | Paper |
|---|---|
| 1. Splits | Section 5.2, Table 1 |
| 2. B0 training, seeds 0-2 | Section 5.4 |
| 3. B3 training, seeds 0-2 | Section 5.4 |
| 4. Exploratory ladder and matcher (seeds 0-2) | Sections 5.5, 6.3 (exploratory cohort) |
| 5. Fresh seeds 3-5 | Section 5.5 |
| 6. Frozen confirmatory matcher analysis | Section 6.3, Table 4 (confirmatory cohort) |
| 7. Grad-CAM faithfulness, random and centre orders | Sections 3.5, 6.6, Tables 6-7, Figs. 6-7 |
| 8. Final tables | Tables 2-7, Supplementary Tables S1-S2 |
| 9. ECE binning sensitivity | Supplementary Table S3 |
| 10. Provenance checks | Section 5.1 |
| 11. Figures | Figs. 2-7, Supplementary Fig. S1 |

## Data

The image datasets are public and must be obtained from their sources:

- MILK10k: ISIC Archive, https://doi.org/10.34970/648456 (CC BY-NC).
- HAM10000: https://doi.org/10.1038/sdata.2018.161
- PAD-UFES-20: https://doi.org/10.1016/j.dib.2020.106221

Section 1 of the notebook reads two split tables (`c3/split.csv`, `c8/ext_split.csv`) that list image paths and harmonised labels for these datasets; the resulting `final_splits.csv` is included in `results/`.

## Derived results (`results/`)

| File | Produced by | Content |
|---|---|---|
| `final_splits.csv` | Section 1 | Image path, label and set for every image |
| `preds_{ce,focal}_s{0..5}.npz` | Sections 2, 5 | B0 logits and labels on every set |
| `preds_b3_{ce,focal}_s{0..5}.npz` | Sections 3, 5 | B3 logits and labels on every set |
| `preds_{ls,mixup}_s{0..2}.npz` | Section 2 | Exploratory B0 variants, not reported in the paper |
| `corr_cal_logits_v2.npz`, `corr_cal_logits_fresh.npz` | Sections 4, 5 | Logits on the corrupted calibration set, seeds 0-2 and 3-5 |
| `l3_rows.csv`, `l3_ens.csv`, `m1_rows.csv` | Section 4 | Exploratory ladder and matcher results |
| `m3_rows.csv` | Section 6 | Confirmatory matcher results |
| `faith_all_models.jsonl` | Section 7 | Per-image deletion/insertion AUCs, Grad-CAM and random order |
| `faith_centre.jsonl` | Section 7 | Per-image deletion/insertion AUCs, centre order |
| `ece_binning.csv` | Section 9 | ECE under four binning schemes (Supplementary Table S3) |

Model weights (`model_*.pt`) are not included because of their size; they are available from the corresponding author on request.

## Rerunning

Copy the contents of `results/` into `/kaggle/working` (or the working directory) and run the analysis cells (Sections 4, 6, 8, 9, 11) on CPU. Section 8 regenerates the table rows (`tables/ladder_rows.csv`, `tables/ladder_ens.csv`) from the `preds_*` files. Retraining (Sections 2, 3, 5) and the Grad-CAM tests (Section 7) need a GPU and the image datasets.

## Contact

Aravinda C. V., Coventry University Kazakhstan, ae7378@coventry.ac.uk
