# 🛡️ DDoS Attack Detection in Software-Defined Networking (SDN) Using Ensemble Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-111111?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![SDN](https://img.shields.io/badge/Domain-Software--Defined%20Networking-0052CC?style=for-the-badge)](https://en.wikipedia.org/wiki/Software-defined_networking)

---

## 📋 Executive Summary

Software-Defined Networking (SDN) decouples network control from data forwarding, offering centralized network management and flexibility. However, this centralized architecture introduces vulnerabilities—most notably, **Distributed Denial of Service (DDoS) attacks** targeting the SDN controller or flow table capacity.

This project implements a robust machine learning framework evaluating **11 distinct classification models**—ranging from traditional algorithms (Logistic Regression, Naive Bayes, SVM, KNN) to advanced deep learning (Artificial Neural Network / MLP) and **Ensemble Learning Techniques** (Bagging, Random Forest, AdaBoost, Gradient Boosting, XGBoost, and Stacking).

By employing a multi-faceted feature selection strategy (Pearson Correlation, Mutual Information, and Principal Component Analysis), the proposed ensemble models achieve up to **100.00% accuracy** and **1.0000 AUC-ROC**, significantly mitigating DDoS attack detection latencies in SDN environments.

---

## 🔍 Key Highlights

- **Dataset**: `ddos_dataset_sdn.csv` containing **104,345 flow entries** with **23 network features**.
- **Data Preprocessing**: Systematic cleaning (handling duplicate records, infinity values, missing data), numeric normalization via `StandardScaler`, and categorical feature encoding via `LabelEncoder`.
- **Hybrid Feature Selection**: Combined relevant feature sets derived from:
  - **Correlation Analysis** (Pearson $|r| > 0.5$)
  - **Mutual Information** (`SelectKBest`, $k=20$)
  - **Principal Component Analysis** (PCA, 20 principal components)
- **Top Performing Models**:
  - 🥇 **Bagging Classifier (Decision Tree Base)**: **100.00% Accuracy**, **1.0000 F1-Score**, **1.0000 AUC**
  - 🥈 **Random Forest Classifier**: **99.995% Accuracy**, **0.99996 F1-Score**, **1.0000 AUC**
  - 🥉 **Simple Stacking Classifier**: **99.968% Accuracy**, **0.99974 F1-Score**, **0.9996 AUC**
  - 🏅 **Artificial Neural Network (ANN/MLP)**: **99.858% Accuracy**, **0.99885 F1-Score**, **0.99999 AUC**
  - 🏅 **Gradient XGBoost**: **99.554% Accuracy**, **0.99638 F1-Score**, **0.99579 AUC**

---

## 🌐 Background: DDoS in SDN Environments

In SDN architecture:
1. **Control Plane**: Centralized controller (e.g., Ryu, POX, OpenDaylight) manages network routing rules.
2. **Data Plane**: OpenFlow switches process incoming packets according to flow table entries.
3. **Attack Vector**: In a DDoS attack, an adversary floods the network with spoofed or unmatched packets. Switches generate `Packet-In` messages to the controller, overwhelming controller bandwidth and filling flow tables.

Detecting malicious flow signatures dynamically near real-time is critical for preserving SDN controller responsiveness.

---

## 📊 Dataset Overview & Features

The raw dataset comprises **104,345 traffic instances** captured across OpenFlow SDN switches:

| Feature Name | Description | Data Type |
| :--- | :--- | :--- |
| `dt` | Time step / epoch duration | Integer |
| `switch` | OpenFlow switch ID | Integer |
| `src` | Source IP Address | Object (Categorical) |
| `dst` | Destination IP Address | Object (Categorical) |
| `pktcount` | Total packet count in the flow entry | Integer |
| `bytecount` | Total byte count transferred | Integer |
| `dur` | Flow duration in seconds | Integer |
| `dur_nsec` | Flow duration in nanoseconds | Integer |
| `tot_dur` | Total flow duration | Float |
| `flows` | Active flow count | Integer |
| `packetins` | Packet-In message rate to SDN controller | Integer |
| `pktperflow` | Average packets per flow | Integer |
| `byteperflow` | Average bytes per flow | Integer |
| `pktrate` | Packet transmission rate | Integer |
| `Pairflow` | Paired flow indicator | Integer |
| `Protocol` | Transport Layer Protocol (`UDP`, `TCP`, `ICMP`) | Object (Categorical) |
| `port_no` | OpenFlow Switch Port Number | Integer |
| `tx_bytes` | Transmitted bytes count | Integer |
| `rx_bytes` | Received bytes count | Integer |
| `tx_kbps` | Transmission rate in kbps | Integer |
| `rx_kbps` | Reception rate in kbps | Float |
| `tot_kbps` | Total bandwidth rate in kbps | Float |
| **`label`** | **Target Class**: `0` (Normal Traffic) / `1` (DDoS Attack Traffic) | Integer |

---

## 🛠️ Data Preprocessing & Pipeline Architecture

```
 Raw SDN Dataset (104,345 rows)
             │
             ▼
   Missing Value & Duplicate Cleaning (99,254 valid rows)
             │
             ▼
   Data Normalization & Encoding (StandardScaler & LabelEncoder)
             │
             ▼
   Hybrid Feature Selection (Correlation + Mutual Info + PCA)
             │
             ▼
   Train / Test Split (80% / 20%)
             │
             ▼
  ┌─────────────────────────────────────────────────────────┐
  │              Model Training & Evaluation                 │
  ├────────────────────────────┬────────────────────────────┤
  │     Ensemble Learning      │   Baseline & Deep ML       │
  │  • Bagging (100.00%)       │  • ANN / MLP (99.86%)      │
  │  • Random Forest (99.99%)  │  • KNN (97.73%)            │
  │  • Stacking (99.97%)       │  • Logistic Reg (76.95%)   │
  │  • XGBoost (99.55%)        │  • SVM (75.35%)            │
  │  • Gradient Boosting       │  • Naive Bayes (66.41%)    │
  │  • AdaBoost                │                            │
  └────────────────────────────┴────────────────────────────┘
```

### 1. Data Cleaning
- Missing values in `rx_kbps` and `tot_kbps` handled alongside duplicate row elimination (`drop_duplicates`).
- Infinite values replaced with `NaN` and dropped.

### 2. Feature Selection Strategy
To optimize computational performance while preserving attack signatures, three techniques were combined via feature union:
- **Pearson Correlation**: Retained features exhibiting $|r| > 0.5$.
- **Mutual Information**: Extracted top 20 features via `SelectKBest(mutual_info_classif, k=20)`.
- **PCA**: Derived top 20 principal components.

---

## 📈 Model Performance & Experimental Results

All models were evaluated on the test dataset using Standard Metrics: **Accuracy**, **Precision**, **Recall**, **F1-Score**, and **Area Under the ROC Curve (AUC)**.

| Rank | Model Classifier | Type | Accuracy (%) | Precision | Recall | F1-Score | AUC-ROC |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **Bagging Classifier (Decision Tree)** | Ensemble | **100.000%** | **1.00000** | **1.00000** | **1.00000** | **1.00000** |
| 🥈 | **Random Forest (RF)** | Ensemble | **99.995%** | **0.99992** | **1.00000** | **0.99996** | **1.00000** |
| 🥉 | **Simple Stacking Classifier** | Ensemble | **99.968%** | **0.99961** | **0.99987** | **0.99974** | **0.99962** |
| 4 | **Artificial Neural Network (ANN/MLP)** | Deep Learning | **99.858%** | **0.99901** | **0.99869** | **0.99885** | **0.99999** |
| 5 | **Gradient XGBoost** | Ensemble | **99.554%** | **0.99802** | **0.99474** | **0.99638** | **0.99579** |
| 6 | **K-Nearest Neighbors (KNN)** | Machine Learning | **97.732%** | **0.97900** | **0.98431** | **0.98165** | **0.97520** |
| 7 | **Gradient Boosting Classifier** | Ensemble | **95.868%** | **0.97074** | **0.96197** | **0.96633** | **0.99233** |
| 8 | **AdaBoost Classifier** | Ensemble | **91.448%** | **0.96745** | **0.89123** | **0.92778** | **0.96544** |
| 9 | **Logistic Regression (LR)** | Baseline | **76.952%** | **0.79624** | **0.84137** | **0.81818** | **0.84261** |
| 10 | **Support Vector Machine (SVM)** | Machine Learning | **75.354%** | **0.77080** | **0.85588** | **0.81111** | **0.82000** |
| 11 | **Gaussian Naive Bayes (GNB)** | Baseline | **66.405%** | **0.72756** | **0.72727** | **0.72741** | **0.74547** |

---

## 🎯 Key Takeaways & Conclusions

1. **Ensemble Models Superiority**: Bagging, Random Forest, and Stacking classifiers substantially outperform single baseline models (LR, SVM, Naive Bayes), achieving near-perfect detection rates.
2. **Low False Alarm Rates**: High precision (>0.999) across ensemble methods ensures minimal false positive alerts for benign network traffic.
3. **Real-Time Feasibility**: Tree-based ensembles deliver swift inference times suitable for integration into SDN controllers for immediate mitigation rules insertion into flow tables.

---

## 💻 Tech Stack & Requirements

- **Language**: Python 3.8+
- **Environment**: Jupyter Notebook / Google Colab
- **Libraries & Dependencies**:
  - `pandas` - Data manipulation & analysis
  - `numpy` - Numerical computing
  - `scikit-learn` - Model building, evaluation metrics, feature selection, scaling
  - `xgboost` - Extreme Gradient Boosting implementation
  - `matplotlib` & `seaborn` - Visualization of ROC curves, confusion matrices, feature importance

---

## 🚀 How to Run the Project

### 1. Clone the Repository
```bash
git clone https://github.com/Bithi769845/Bithi_DDoS_Attack_Detction_for_SDN.git
cd Bithi_DDoS_Attack_Detction_for_SDN
```

### 2. Install Required Packages
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn jupyter
```

### 3. Prepare the Dataset
Ensure `ddos_dataset_sdn.csv` is placed in your working directory or update the file path inside the notebook:
```python
fp = 'path/to/ddos_dataset_sdn.csv'
```

### 4. Open and Execute Notebook
Launch Jupyter Notebook:
```bash
jupyter notebook Bithi_DDoS_Attack_Detction_for_SDN.ipynb
```
Execute cells sequentially to replicate preprocessing, feature selection, model training, and ROC visualizations.

---

## 📁 Repository Structure

```
Bithi_DDoS_Attack_Detction_for_SDN/
├── Bithi_DDoS_Attack_Detction_for_SDN.ipynb  # Primary Jupyter notebook with complete pipeline
└── README.md                                 # Project documentation
```

---

## 📝 Citation & Contact

If you use this work or findings in your research, please consider citing or referencing this repository:
```
@misc{ddos_sdn_ensemble_2026,
  author = {Bithi},
  title = {DDoS Attack Detection in SDN using Ensemble Machine Learning Techniques},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/Bithi769845/Bithi_DDoS_Attack_Detction_for_SDN}}
}
```
