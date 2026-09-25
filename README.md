# Customer Support Ticket Intelligence System 🎫⚡

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Zero External APIs](https://img.shields.io/badge/APIs-100%25%20Offline-emerald.svg)]()
[![Test Suite](https://img.shields.io/badge/tests-12%20passed-brightgreen.svg)]()
[![Accuracy](https://img.shields.io/badge/Test%20Accuracy-95.0%25-blueviolet.svg)]()

An end-to-end, machine-learning-powered **Customer Support Ticket Classification and Automated Response System** built with Python and Scikit-Learn.

> **Zero External APIs**: Operates 100% locally and offline without external paid services or cloud API keys.

---

## 📌 Project Overview & Problem Statement

Enterprises receive high volumes of unstructured customer support inquiries across chat, email, and web portals. Manual ticket sorting causes triage delays, human misrouting, and increased operational costs.

This project delivers a complete AI/ML solution that:
1. Automatically ingests raw ticket descriptions.
2. Cleans and normalizes noisy customer text.
3. Classifies tickets into **8 operational categories** with **95.0% test accuracy**.
4. Generates probabilistic confidence scores and class distributions.
5. Produces immediate, context-aware automated customer resolution responses (Bonus GenAI task).
6. Provides both an **interactive Command-Line Interface (CLI)** and a **modern FastAPI Web Interface**.

---

## 🛠️ Technologies Used

- **Programming Language**: Python 3.11
- **Data Manipulation & Analysis**: Pandas, NumPy
- **Machine Learning & NLP**: Scikit-Learn (`TfidfVectorizer`, `LogisticRegression`, `CalibratedClassifierCV`, `MultinomialNB`, `RandomForestClassifier`), Joblib
- **Data Visualization**: Matplotlib, Seaborn
- **Web Application & REST API**: FastAPI, Uvicorn, Pydantic
- **Frontend**: Modern Vanilla HTML5, CSS3 (Glassmorphism & responsive grid), Vanilla JavaScript
- **Testing & Verification**: PyTest, Starlette TestClient

---

## 📊 Dataset Information

- **Dataset File**: `data/customer_support_ticket_dataset_200.csv`
- **Total Records**: 200 customer support tickets
- **Columns**: `ticket_id`, `customer_name`, `ticket_description`, `date`, `priority`, `status`, `category`
- **Class Balance**: Exactly 8 balanced categories (25 tickets each, 12.5%):
  1. `Login Issue`
  2. `Application Error`
  3. `Report`
  4. `Account Update`
  5. `Performance`
  6. `Payment Issue`
  7. `Access Issue`
  8. `Data Issue`
- **Quality**: 0 missing values, 0 duplicate rows.

---

## 📂 Project Directory Structure

```
support-ticket-intelligence/
├── data/
│   └── customer_support_ticket_dataset_200.csv   # Operational dataset (200 records)
├── models/
│   └── ticket_classifier.joblib                  # Serialized ML pipeline artifact
├── screenshots/
│   ├── category_distribution.png                 # EDA category chart
│   ├── priority_distribution.png                 # EDA priority breakdown
│   ├── text_length_distribution.png              # EDA word count distribution
│   ├── confusion_matrix.png                      # Test evaluation heatmap
│   └── web_interface.png                         # Verified web UI screenshot
├── src/
│   ├── __init__.py
│   ├── preprocessing.py                          # Text cleaning & validation pipeline
│   ├── eda.py                                    # EDA analysis & chart generation
│   ├── train.py                                  # Model benchmarking, training & eval
│   ├── predict.py                                # Inference pipeline & test benchmarks
│   ├── responder.py                              # Offline automated customer response
│   ├── cli.py                                    # Interactive command-line app (Option A)
│   └── app.py                                    # FastAPI web server (Option B)
├── static/
│   ├── index.html                                # Web application frontend
│   ├── style.css                                 # Glassmorphism design system
│   └── app.js                                    # Dynamic UI & fetch handling
├── tests/
│   └── test_pipeline.py                          # 12 automated unit tests
├── requirements.txt                              # Clean dependency specification
├── SOLUTION_REPORT.md                            # Comprehensive Task 7 technical report
├── .gitignore                                    # Git ignore rules
└── README.md                                     # Project documentation
```

---

## 🚀 Installation & Quickstart

### 1. Clone Repository & Setup Virtual Environment
```bash
git clone <your-repository-url>
cd support-ticket-intelligence

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 📈 Running the Pipeline

### Step 1: Exploratory Data Analysis (EDA)
Inspect dataset statistics and generate visualizations in `screenshots/`:
```bash
python src/eda.py
```

### Step 2: Model Training & Evaluation
Benchmarks 4 candidate algorithms with 5-fold cross-validation, trains the champion Logistic Regression model on an 80/20 stratified split, generates `screenshots/confusion_matrix.png`, and saves `models/ticket_classifier.joblib`:
```bash
python src/train.py
```

**Evaluation Results**:
```
Accuracy           : 95.00%
Macro Precision    : 96.43%
Macro Recall       : 95.00%
Macro F1-Score     : 94.79%
Weighted F1-Score  : 94.79%
```

### Step 3: Run Prediction & Unseen Benchmarks
Run the automated 8-case benchmark suite on unseen tickets:
```bash
python src/predict.py --test
```

Predict a custom single ticket description:
```bash
python src/predict.py --text "I forgot my password and cannot login" --priority "High"
```

---

## 💻 User Interfaces (Task 6)

### Option A: Interactive Command Line Interface (CLI)
Start an interactive terminal session:
```bash
python src/cli.py
```

**Interactive Example**:
```
Enter Ticket Description > The application is taking too long to load and times out.

------------------------------------------------------------
PREDICTED CATEGORY : [PERFORMANCE]
CONFIDENCE SCORE   : 37.1%
RECOMMENDED ACTION : Infrastructure scaling and database query profiling.
ESTIMATED SLA      : Within 24 business hours (Standard SLA)
------------------------------------------------------------
```

### Option B: Modern Web Application
Start the local FastAPI web server:
```bash
python src/app.py
```
Open your browser and navigate to:
👉 **`http://127.0.0.1:8000`**

#### Web UI Features:
- 🌟 Glassmorphic dark theme with vibrant category accents.
- ⚡ 1-click sample preset chips for fast testing.
- 📊 Real-time confidence gauge and 8-category probability distribution.
- 💬 100% offline automated customer response with one-click copy.
- ⏱️ Operational SLA estimation and recommended agent triage actions.

---

## 🧪 Automated Testing

Execute the complete test suite verifying text preprocessing, model inference, offline responses, and FastAPI endpoints:
```bash
pytest tests/test_pipeline.py -v
```
*(All 12 unit tests pass).*

---

## 📄 Solution Report

For deep architectural explanations, mathematical formulation of TF-IDF, model comparison tables, confusion matrix analysis, system limitations, and future enterprise enhancements, see:
👉 [**SOLUTION_REPORT.md**](SOLUTION_REPORT.md)

---

## 🔒 Security & Privacy Notice
This project uses **zero external API calls**. No data, tickets, or user details are transmitted over the internet or sent to third-party AI providers. All computation remains 100% local.
