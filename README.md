# Bank Marketing Term Deposit Predictor

> A predictive machine learning system that optimizes telemarketing targeting by forecasting client term deposit subscriptions, leveraging a banking dataset of 41,188 data points across 20 features.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)

---

## 📌 1. About the Project

This project focuses on building a highly accurate binary classification model to predict whether a customer will subscribe (`yes` or `no`) to a bank term deposit based on their interaction history with retail marketing campaigns. 

By analyzing comprehensive institutional and macroeconomic features, the project evaluates multiple supervised learning algorithms to find the absolute balance between prediction accuracy and target recall—maximizing institutional conversion rates while drastically minimizing cold-call overhead.

### 🛠️ Built With & Tech Stack
* **Core Language:** Python 3
* **Data Processing & Analytics:** Pandas, NumPy
* **Data Visualization:** Seaborn, Matplotlib
* **Machine Learning Pipelines:** Scikit-Learn (Supervised Classification algorithms, Preprocessing modules, and Evaluation Metrics)


### 📊 Dataset Source
The data used in this project is the **Bank Marketing Dataset**. Due to GitHub file size recommendations and to keep the repository clean, the raw data file is hosted externally.

* 📥 **Download Link:** You can access and download the raw dataset here: [Google Drive Dataset Link](https://drive.google.com/file/d/1Kw3bRgOach2sEbAAtBXeZfIoUYzsJ2TJ/view?usp=sharing)
* ⚙️ **Setup Instruction:** Download the `Bank Marketing.csv` file from the link above and place it inside your local project directory (or create a `Data/` folder) before executing the Jupyter Notebook.
  
---

## 💡 2. Problem & Value Proposition

### The Problem
Traditional financial telemarketing is highly inefficient. Direct-marketing campaigns heavily rely on bulk outbound calling strategies that yield low conversion rates. This structural approach creates a two-fold business loss: excessive operations budgets are burned on uncooperative leads, and repeated contact attempts increase customer fatigue, damaging overall brand sentiment.

### The Solution
By applying machine learning pipelines to historical interaction datasets, this project builds a smart filtering engine. This enables banking institutions to:
* **Target with Precision:** Segment and reach out only to customers with a high probability of institutional conversion.
* **Optimize Operational Expenditures:** Allocate calling agents efficiently, bypassing dead-leads and reducing cost per acquisition (CPA).
* **Maximize Capital Inflow:** Identify high-value cohorts to accelerate the growth of long-term retail deposit accounts.

### ⚙️ Feature Complexity Breakdown
The system processes 21 distinct structural parameters, classified under three structural branches:
1. **Customer Demographics:** Age, Job, Marital Status, Education, Financial Defaults, Housing/Personal Loans.
2. **Campaign Touchpoints:** Contact Method, Communication Month/Day, Previous Contact Outcome, Contact Duration, and Attempt Intervals.
3. **Macroeconomic Indicators:** Employment Variance Rate, Consumer Price Index (CPI), Consumer Confidence Index (CCI), Euribor 3-Month Interest Rates, and Number of Employees.

---

## 📈 3. Data Preprocessing & Model Performance

### Data Engineering Pipeline
To ensure high-fidelity model training, the following data pipeline operations were structured into the source code:
* **Categorical Feature Encoding:** Managed 11 discrete categorical variables using systematic label/one-hot transformations to maintain structural data integrity.
* **Feature Correlation Analysis:** Constructed an exploratory $N \times N$ Pearson correlation matrix map to isolate and review co-dependencies among quantitative features.
* **Handling Class Imbalance:** Factored inherent marketing data imbalances (where "no" responses heavily outnumber structural "yes" transitions) to protect algorithms from defaulting to majority-class biases.

### Model Comparison & Performance Ledger
The performance of the classification models was thoroughly validated across critical business metrics:

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Neural Network (MLP)** | **High** | High | High | Balanced | **Top Tier** |
| **K-Nearest Neighbors (KNN)** | High | High | Moderate | Balanced | High |
| **Support Vector Machine (SVM)** | Robust | Robust | Robust | Robust | Robust |
| **Decision Tree** | High | Moderate | Moderate | Moderate | Moderate |
| **Random Forest** | High | High | Moderate | Balanced | High |
| **Logistic Regression** | Baseline | Baseline | Baseline | Baseline | Baseline |
| **Naive Bayes** | ~20% | Low | **~1.00** | Low | Low |

### 🛠️ Strategic Engineering Takeaways
1. **Non-Linear Dominance:** Non-linear structural frameworks (Neural Networks and KNN) significantly outperformed linear baselines, successfully identifying complex interplay paths across the 21 unique dimensional attributes.
2. **The Naive Bayes Recall Trade-off:** Naive Bayes displayed near perfect Recall (~1.0) but suffered from critically low Precision and Accuracy (~20%) due to an aggressive over-prediction of the positive class, providing a clear case study on managing class-imbalance anomalies.
3. **The 'Duration' Caveat:** Feature correlation reviews highlight that call `duration` is an exceptionally strong predictive weight. However, because its value is only finalized *after* a campaign call completes, its influence must be carefully controlled in live, real-time predictive deployments to prevent optimistic telemetry distortion.


[📖 Read the Full Technical Report](./project_report.pdf)
---

## 🚀 4. Getting Started & Implementation

### Prerequisites
Ensure your local system runs Python 3.8+ along with the required workspace libraries:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter

```

### Repository Structure

```filepath
├── Data/
│   └── Bank Marketing.csv     # Target raw tabular dataset (41,188 x 21)
├── Bank_Marketing.ipynb       # Core development notebook with EDA & training loops
├── project_report.pdf         # Technical compilation and structural analysis report
└── README.md                  # System overview and recruiter guide

```

### Local Execution Strategy

1. **Clone the Repository:**
```bash
git clone https://github.com/SakibMahmud111/bank-term-deposit-predictor.git
cd bank-term-deposit-predictor

```


2. **Launch the Notebook:** Run the development file inside your local Jupyter or Google Colab workspace environment:
```bash
jupyter notebook Bank_Marketing.ipynb

```
