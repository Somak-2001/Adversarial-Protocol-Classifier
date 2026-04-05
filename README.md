# Adversarial ML: Attacking & Defending Protocol Classifiers

> **Term Project — CS60099: Internet and Applications**
> Department of Computer Science and Engineering, IIT Kharagpur

---

## Overview

This project investigates the vulnerability of machine learning-based network protocol classifiers to adversarial attacks and evaluates defense mechanisms to improve model robustness.

A **Random Forest classifier** is trained on the [CICIDS2017](https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset) network traffic dataset to classify network flows into benign and attack categories. Two adversarial evasion attacks — **FGSM** and **PGD** — are adapted for the non-differentiable ensemble model using gradient-free approximations. Two defenses — **adversarial training** and **input sanitization** — are evaluated against these attacks.

---

## Results Summary

| Condition | Accuracy (%) | Accuracy Drop (%) | ASR (%) |
|---|---|---|---|
| Baseline (Clean) | 99.76 | — | — |
| FGSM Attack (ε=0.1) | 84.36 | 15.40 | 15.53 |
| PGD Attack (ε=0.1) | 87.87 | 11.89 | 12.02 |
| Input Sanitization | 87.68 | 12.08 | — |
| Adversarial Training | 99.37 | 0.39 | — |

---

## Project Structure

```
adversarial-protocol-classifier/
│
├── IAP_TermProject.ipynb          # Main notebook (full pipeline)
├── report/
│   └── adversarial_ml_paper.pdf   # Term paper (LaTeX compiled)
├── figures/
│   ├── baseline_cm_clean.png
│   ├── fgsm_attack_cm.png
│   ├── defense_cm.png
│   ├── model_robustness_bar.png
│   ├── fgsm_epsilon_vs_accuracy.png
│   └── defense_robustness_vs_attack_strength.png
├── README.md
└── .gitignore
```

---

## Dataset

**CICIDS2017** — Canadian Institute for Cybersecurity Intrusion Detection Dataset 2017

- **Source:** [Kaggle — Network Intrusion Dataset](https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset)
- **Size:** ~2.83 million flow records, 78 features, 15 traffic classes
- **Features:** Flow duration, packet length stats, inter-arrival times, flag counts, throughput metrics
- **Classes:** BENIGN, DoS Hulk, PortScan, DDoS, FTP-Patator, SSH-Patator, Bot, Web Attacks, and more

> **Note:** The dataset is not included in this repository due to size. Download it from the Kaggle link above and place the CSV files in a `/dataset` folder before running the notebook.

---

## Methodology

### Preprocessing
- Combined all 8 CSV files (~2.83M rows, 79 columns)
- Removed NaN and infinite values
- Sampled 200,000 records for computational efficiency
- Label encoding, StandardScaler normalization, 80:20 train-test split

### Model
- **Random Forest** — 100 estimators, `random_state=42`, `n_jobs=-1`

### Adversarial Attacks
Since Random Forest is **non-differentiable**, gradient-based attacks (FGSM, PGD) are approximated using **gradient-free random sign perturbations**, simulating a realistic black-box threat scenario.

| Attack | Type | Epsilon | Steps |
|---|---|---|---|
| FGSM | Single-step | 0.1 | 1 |
| PGD | Iterative | 0.1 | 10 (α=0.01) |

### Defense Mechanisms
- **Adversarial Training** — Retrained on clean + adversarially perturbed samples (2× training data)
- **Input Sanitization** — Feature clipping to training data min/max bounds

---

## Setup & Usage

### Requirements
```
python >= 3.10
scikit-learn
numpy
pandas
matplotlib
seaborn
```

### Running the Notebook

1. Clone the repository:
```bash
git clone https://github.com/<your-username>/adversarial-protocol-classifier.git
cd adversarial-protocol-classifier
```

2. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset) and extract CSVs into `/dataset`

3. Open and run the notebook:
```bash
jupyter notebook IAP_TermProject.ipynb
```

> The notebook was originally developed on **Google Colab**. If running locally, remove the `drive.mount` and `!unzip` cells and update the dataset path accordingly.

---

## Group Members

| Name | Roll No. |
|---|---|
| Somak Poddar | 25CS60R12 |
| Aarobh | M.Tech, CSE |
| Karthik | M.Tech, CSE |

**Course:** CS60099 — Internet and Applications
**Institute:** IIT Kharagpur

---

## References

1. Goodfellow et al., *Explaining and Harnessing Adversarial Examples*, ICLR 2015
2. Madry et al., *Towards Deep Learning Models Resistant to Adversarial Attacks*, ICLR 2018
3. Sharafaldin et al., *Toward Generating a New Intrusion Detection Dataset*, ICISSP 2018
4. Papernot et al., *Practical Black-Box Attacks against Machine Learning*, ASIACCS 2017
5. Pedregosa et al., *Scikit-learn: Machine Learning in Python*, JMLR 2011
