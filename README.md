# Adversarial Machine Learning: Attacking & Defending Protocol Classifiers

> **Term Project — CS60008: INTERNET ARCHITECTURE AND PROTOCOLS**  
> **Department of Computer Science and Engineering, Indian Institute of Technology Kharagpur**  
> **Topic 9: Adversarial ML: Attacking & Defending Protocol Classifiers**

---

## Executive Summary

Machine learning models deployed in network monitoring, quality-of-service (QoS) routing, and Software-Defined Networking (SDN) architectures are frequently tasked with **network protocol classification**—identifying underlying transport and application protocols governing network flows. However, statistical classifiers are inherently susceptible to **adversarial evasion attacks**: small, maliciously crafted feature-space perturbations that induce severe misclassifications at inference time while preserving functional traffic utility.

This repository provides an end-to-end, reproducible research pipeline implementing and evaluating:
1. **Multiclass Network Protocol Classification** using the **UNSW-NB15** dataset (`proto` target).
2. **Leakage-Free Preprocessing**: Complete removal of target leakage (`Label`, `attack_cat`) and topological/proxy shortcuts (`srcip`, `dstip`, `sport`, `dsport`, `Stime`, `Ltime`).
3. **Four Diverse Model Architectures**:
   - **Random Forest** (non-differentiable ensemble)
   - **Linear SVM** (scalable linear convex boundary)
   - **1D Convolutional Neural Network (CNN)** (differentiable deep feature extractor)
   - **Long Short-Term Memory (LSTM)** (sequential feature processor with explicit tabular limitation documentation)
4. **Real Gradient-Based Evasion Attacks**:
   - **True FGSM (Fast Gradient Sign Method)**: Exact gradient-directed perturbation $\epsilon \cdot \text{sign}(\nabla_x \mathcal{L})$.
   - **True PGD (Projected Gradient Descent)**: Iterative projected gradient ascent inside the $L_\infty$ ball.
   - **Feature-Space Invariance Masking**: Perturbations strictly isolated to continuous flow features; categorical one-hot indicators remain invariant.
   - **Transfer-Based Black-Box Attack**: Evaluating gradient-derived adversarial examples from surrogate 1D CNN against gradient-inaccessible Random Forest and Linear SVM.
5. **Defense Mechanisms**:
   - **Adversarial Training**: Augmented min-max optimization retraining on clean and gradient-perturbed flow samples (50/50 mix).
   - **Input Sanitization**: Baseline defense clipping standardized continuous features to $[-3, +3]$.
6. **Programmatic Evaluation**: Dynamic generation of Tables 1, 2, and 3, confusion matrices, learning curves, and robustness trade-off plots.

---

## Repository Structure

```
Adversarial-Machine-Learning/
│
├── IAP_AdversarialML.ipynb    # Main end-to-end self-contained notebook (Colab-ready)
├── README.md                  # Comprehensive project documentation
├── assignment.txt             # Original assignment requirements and topic description
│
└── dataset/
    └── UNSW-NB15/             # Dataset directory
        ├── NUSW-NB15_features.csv   # 49-column feature schema definition
        ├── UNSW-NB15_1.csv          # Raw flow records (shard 1)
        ├── UNSW-NB15_2.csv          # Raw flow records (shard 2)
        ├── UNSW-NB15_3.csv          # Raw flow records (shard 3)
        ├── UNSW-NB15_4.csv          # Raw flow records (shard 4)
        └── unsw-nb15-complete.zip   # Pre-compressed dataset archive (~143 MB)
```

---

## Experimental Settings & Configuration

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Dataset** | UNSW-NB15 | Raw flow records across 4 CSV shards |
| **Classification Target** | `proto` | Multiclass transaction protocol (e.g., TCP, UDP, ARP, OSPF, ICMP, SCTP) |
| **Random Seed** | `42` | Enforced across sampling, train/test splitting, and model initializations |
| **Shard Sampling** | 50,000 / shard | Memory-efficient chunked reservoir sampling (`chunksize=100,000`) |
| **Class Balancing** | Min 20, Max 1,500 | Protocol frequency threshold and per-protocol quota cap |
| **Train / Test Split** | 80% / 20% | Stratified split executed **before** fitting any preprocessor |
| **Continuous Features** | 38 features | Imputed with median and scaled with `StandardScaler` on `X_train` |
| **Categorical Features**| 2 (`state`, `service`) | Imputed with mode and encoded with `OneHotEncoder` on `X_train` |
| **Attack Space** | $L_\infty$ ball | Standardized continuous feature space; categorical features masked |
| **FGSM Budgets ($\epsilon$)**| `[0.01, 0.05, 0.10, 0.20]` | Multi-epsilon evaluation grid |
| **PGD Configuration** | $\epsilon=0.10, \alpha=0.02, K=10$ | 10-step iterative projected gradient ascent |
| **Adversarial Training** | $\epsilon=0.10, 12$ epochs | 50/50 mixture of clean and FGSM adversarial samples |
| **Input Sanitization** | $[-3.0, +3.0]$ | Heuristic clipping of standardized continuous feature values |

---

## Methodology

### 1. Leakage-Free Preprocessing Pipeline
To prevent data leakage and ensure authentic learning:
1. **Target Isolation**: Target is strictly `proto` (Column No. 5).
2. **Leakage Dropped**: Intrusion labels (`attack_cat`, `Label`) are removed.
3. **Shortcuts Dropped**: Identifiers (`srcip`, `dstip`), port proxies (`sport`, `dsport`), and timestamp artifacts (`Stime`, `Ltime`) are removed.
4. **Data Partitioning**: Stratified 80:20 train/test split is applied directly to raw data before transformer fitting.
5. **Feature Invariance Masking**: We construct a binary feature mask $\mathbf{m} \in \{0, 1\}^D$ where $m_j = 1.0$ for continuous numerical metrics and $m_j = 0.0$ for one-hot categorical indicators, ensuring categorical features are preserved during adversarial attacks.

### 2. Model Architectures
- **Random Forest**: 100 estimators, balanced subsample weighting to handle remaining multiclass protocol imbalance.
- **Linear SVM**: `LinearSVC` with balanced weighting, providing a scalable linear convex boundary.
- **1D CNN**: Input `(B, 1, D)` $\to$ Conv1D(32) $\to$ ReLU $\to$ MaxPool1D $\to$ Conv1D(64) $\to$ ReLU $\to$ AdaptiveMaxPool1D $\to$ Dense(64) $\to$ Dropout(0.25) $\to$ Output($C$). Trained with Adam and early stopping.
- **LSTM**: Input `(B, D, 1)` $\to$ LSTM(64) $\to$ Dropout(0.25) $\to$ Output($C$). *(Explicitly documented: tabular flow records represent summary statistics, not sequential packet arrivals; LSTM serves as an architectural benchmark)*.

### 3. Adversarial Threat Model & Attacks
- **Threat Model**: Inference-time evasion attack modifying standardized flow feature vectors within an $L_\infty$ bound $\|\mathbf{x}_{adv} - \mathbf{x}\|_\infty \le \epsilon$.
- **True FGSM**:
  $$\mathbf{x}_{adv} = \mathbf{x} + \epsilon \cdot \text{sign}\left(\nabla_\mathbf{x} \mathcal{L}(f_\theta(\mathbf{x}), y)\right) \odot \mathbf{m}$$
- **True PGD**:
  $$\mathbf{x}^{(t+1)} = \Pi_{\mathbf{x} + \mathcal{S}_\epsilon} \left( \mathbf{x}^{(t)} + \alpha \cdot \text{sign}\left(\nabla_{\mathbf{x}^{(t)}} \mathcal{L}(f_\theta(\mathbf{x}^{(t)}), y)\right) \odot \mathbf{m} \right)$$
- **Attack Success Rate (ASR)**: Strictly calculated over samples initially correctly classified on clean data:
  $$\text{ASR} = \frac{\sum ((\hat{y}_{clean} == y) \;\&\; (\hat{y}_{adv} \ne y))}{\sum (\hat{y}_{clean} == y)} \times 100\%$$
- **Transfer-Based Black-Box Attack**: Generates adversarial examples using the differentiable 1D CNN surrogate and queries the gradient-inaccessible Random Forest and Linear SVM targets.

### 4. Defense Mechanisms
- **Adversarial Training**: Min-max optimization retraining the 1D CNN on a dynamic 50/50 mixture of clean and FGSM-perturbed training batches.
- **Input Sanitization**: Simple baseline heuristic clipping continuous features to $[-3.0, +3.0]$ standard deviations.

---

## Google Colab Execution Guide

The notebook is optimized for **Google Colab** with GPU acceleration:

### Step 1: Open Colab & Upload Notebook
1. Open [Google Colab](https://colab.research.google.com/).
2. Click **Upload** and select `IAP_AdversarialML.ipynb`.

### Step 2: Enable GPU Accelerator
1. In Colab menu: **Runtime** $\to$ **Change runtime type**.
2. Under **Hardware accelerator**, select **T4 GPU** and click **Save**.

### Step 3: Upload Dataset to Session Storage
1. On the left sidebar, click the **Folder icon 📁** (Files panel).
2. Click the **Upload to session storage icon** (or drag and drop):
   - Select `dataset/UNSW-NB15/unsw-nb15-complete.zip` (~143 MB) from your local project folder.
   - It will upload directly to `/content/unsw-nb15-complete.zip`.

### Step 4: Run All Cells
1. In Colab menu: **Runtime** $\to$ **Run all** (or press `Ctrl + F9`).
2. The notebook will automatically:
   - **Cell 2**: Detect and unzip `unsw-nb15-complete.zip` into `/content/dataset/UNSW-NB15/` in ~3 seconds.
   - **Cell 4 & 7**: Auto-discover the dataset, perform randomized sampling, and balance protocol quotas.
   - **Cells 13–19**: Train and evaluate Random Forest, Linear SVM, 1D CNN, and LSTM.
   - **Cell 21**: Generate **TABLE 1** (Clean Model Comparison) and Figure 1.
   - **Cells 26 & 28**: Execute white-box FGSM & PGD attacks and transfer-based black-box attacks against RF & SVM.
   - **Cell 30**: Generate **TABLE 2** (Adversarial Attack Performance) and Figures 2 & 3.
   - **Cells 32 & 34**: Train and evaluate Adversarial Training and Input Sanitization defenses.
   - **Cell 36**: Generate **TABLE 3** (Defense Effectiveness Comparison) and Figures 4 & 5.

---

## Programmatic Output Summary

The notebook computes and renders all results programmatically without hardcoded or fabricated numbers:

- **TABLE 1: Clean Baseline Model Comparison**
  - Evaluates: Random Forest, Linear SVM, 1D CNN, LSTM
  - Metrics: Accuracy (%), Macro Precision, Macro Recall, Macro F1, Weighted F1, Training Time (s), Inference Time (s)
- **TABLE 2: Adversarial Attack Results (White-Box & Black-Box)**
  - Evaluates: 1D CNN (White-Box FGSM across $\epsilon$, PGD), Random Forest (Transfer), Linear SVM (Transfer)
  - Metrics: Perturbed Accuracy (%), Accuracy Degradation (%), Attack Success Rate (ASR %), Mean $L_\infty$, Mean $L_2$
- **TABLE 3: Defense Mechanisms Effectiveness Comparison**
  - Evaluates: Baseline 1D CNN, Adversarially Trained CNN, Input Sanitization (Clipping to $[-3, +3]$)
  - Metrics: Clean Accuracy (%), FGSM Accuracy (%), PGD Accuracy (%), FGSM ASR (%), PGD ASR (%), Robustness Gain (%)
- **Visualizations**:
  - Figure 1: Clean performance comparison across models (Accuracy, Macro F1, Weighted F1)
  - Figure 2: Perturbation budget ($\epsilon$) vs. Model Accuracy and Attack Success Rate (ASR)
  - Figure 3: Confusion matrix of 1D CNN under White-Box PGD attack
  - Figure 4: Defense comparison bar chart under clean and attacked conditions
  - Figure 5: Confusion matrix of Adversarially Trained CNN under PGD attack

---

## Methodological Insights & Limitations

1. **Feature Space vs. Problem Space**:
   - The attacks perturb standardized tabular flow metrics, representing a theoretical upper-bound vulnerability. In live network deployments, creating equivalent packets requires solving the problem-space mapping problem to satisfy protocol state machines and checksums.
2. **Tabular Sequence Representation**:
   - Treating tabular flow statistics as sequential inputs for LSTM is an architectural benchmark configuration; flow summaries do not represent sequential temporal packet bursts.
3. **Protocol Class Imbalance**:
   - Realistic traffic is heavily skewed towards TCP and UDP. Stratified quota sampling guarantees minority protocol evaluation, but operational deployments benefit from hierarchical or anomaly-based detection for rare protocols.

---

## Group Members

| Name | Roll No. | Department / Program |
| :--- | :--- | :--- |
| **Somak Poddar** | 25CS60R12 | M.Tech, Computer Science and Engineering |
| **Aarobh** | — | M.Tech, Computer Science and Engineering |
| **Karthik** | — | M.Tech, Computer Science and Engineering |
| **Karthik** | — | Dual Degree, Computer Science and Engineering |

**Course**: CS60099 — Internet and Applications  
**Institute**: Indian Institute of Technology Kharagpur  

---

## Academic References

1. Goodfellow, I. J., Shlens, J., & Szegedy, C. (2014). *Explaining and harnessing adversarial examples*. arXiv preprint arXiv:1412.6572.
2. Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2018). *Towards deep learning models resistant to adversarial attacks*. International Conference on Learning Representations (ICLR 2018).
3. Papernot, N., McDaniel, P., & Goodfellow, I. (2017). *Practical black-box attacks against machine learning*. In Proceedings of the 2017 ACM on Asia Conference on Computer and Communications Security (Asia CCS), pp. 506–519.
4. Usama, M., et al. (2019). *Examining machine learning for network traffic classification: An adversarial perspective*. IEEE Transactions on Network and Service Management.
5. Pierazzi, F., Pendlebury, F., Cortellazzi, J., & Cavallaro, L. (2020). *Intriguing properties of adversarial ML attacks in the problem space*. In 2020 IEEE Symposium on Security and Privacy (SP), pp. 1332–1349.
6. Meng, D., & Chen, H. (2017). *MagNet: a two-pronged defense against adversarial examples*. In Proceedings of the 2017 ACM SIGSAC Conference on Computer and Communications Security (CCS), pp. 135–147.
7. Moustafa, N., & Slay, J. (2015). *UNSW-NB15: a comprehensive data set for network intrusion detection systems*. In 2015 Military Communications and Information Systems Conference (MilCIS), pp. 1–6.
