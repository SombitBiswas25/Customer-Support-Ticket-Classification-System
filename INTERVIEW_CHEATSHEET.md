# 🎯 Customer Support Ticket Intelligence — Interview Cheat Sheet

Keep this open during your interview. It has exact answers, metrics, and architecture points.

---

## ⚡ 1. The Quick Numbers & Facts (Glance Sheet)

| Parameter | Value |
| :--- | :--- |
| **Project Name** | Customer Support Ticket Intelligence System |
| **Core Problem** | Automated multi-class routing & instant response for customer support tickets |
| **Model Used** | **Multinomial Logistic Regression** with $L_2$ Regularization |
| **Feature Extraction**| **TF-IDF Vectorizer** (Unigrams + Bigrams, Sublinear TF, 1000 max features) |
| **Dataset Size** | **200 records**, **8 perfectly balanced categories** (25 tickets each, 12.5%) |
| **Train / Test Split**| **80% Train (160) / 20% Test (40)** — **Stratified** |
| **Holdout Test Accuracy** | **95.00%** (38/40 test tickets correctly classified) |
| **Macro Precision / Recall / F1** | **Precision: 96.43%** \| **Recall: 95.00%** \| **Macro F1: 94.79%** |
| **Unseen Benchmark Suite** | **8 / 8 Correct (100.0%)** |
| **Key Innovation** | **Confidence Threshold Fallback** (catches vague tickets < 40% & routes to human triage) |
| **Tech Stack** | Python 3.11, Scikit-Learn, Pandas, NumPy, FastAPI, Uvicorn, Vanilla HTML/CSS/JS |
| **Privacy / Cost** | **100% Offline / Zero External APIs** (No OpenAI, no cloud latency, GDPR-safe) |

---

## 🗣️ 2. The Project Pitches

### A. The 30-Second Elevator Pitch
> *"I designed and built an end-to-end, 100% offline Machine Learning system that classifies incoming customer support tickets into 8 business categories and generates immediate, automated resolution responses.*
> *Using a TF-IDF bi-gram pipeline and Multinomial Logistic Regression, it delivers **95% accuracy** and a **94.8% Macro F1-score** with sub-2ms inference time. I also implemented an operational confidence threshold fallback that detects ambiguous tickets and routes them to human triage."*

### B. The 2-Minute Walkthrough
1. **The Business Problem**: In enterprise support, thousands of raw text tickets arrive daily. Manual triage causes 24–48 hour bottlenecks and misrouting.
2. **The Constraints**: Enterprise data privacy (zero PII sent to cloud APIs) and zero per-token API costs.
3. **The Solution Pipeline**:
   - **Data Cleaning**: Stripped regex noise, URLs, emails, special symbols, and folded to lowercase.
   - **Feature Extraction**: TF-IDF with unigrams and bigrams to capture phrases like *"cannot login"* or *"access denied"*.
   - **Model Benchmarking**: Tested Logistic Regression, Calibrated LinearSVC, Naive Bayes, and Random Forest using 5-Fold Stratified Cross-Validation. Logistic Regression achieved **96.88% CV accuracy** and was selected as champion.
   - **Inference & UI**: Served via a FastAPI backend powering a glassmorphic dashboard with live confidence meters, class distributions, and automated customer replies.

---

## 🛠️ 3. "Why Did You Choose...?" (Technical Justifications)

### Q1: Why TF-IDF with $(1, 2)$ N-grams?
* *"Unigrams alone lose critical negation and compound context. For example, `'login'` is benign, but `'cannot login'` is an urgent issue. Bi-grams preserve compound phrases. We used sublinear TF scaling ($1 + \log(TF)$) to dampen the impact of frequently repeated words."*

### Q2: Why Logistic Regression over Random Forest or Naive Bayes?
* *"Three concrete reasons:*
  1. *Text classification vectors are **sparse and high-dimensional**; linear hyperplanes excel in this geometry.*
  2. *Softmax outputs **well-calibrated probabilities** ($0\%$ to $100\%$), which directly power our confidence scores and triage thresholds.*
  3. *Inference takes **< 2 milliseconds** on standard CPU, with a deterministic, lightweight footprint."*

### Q3: Why a Stratified Split?
* *"In multi-class classification, random splitting can accidentally leave minority categories with too few test samples. A **Stratified 80/20 split** ensures exactly 5 test tickets per category across all 8 classes, giving statistically rigorous evaluation."*

### Q4: Why no external LLM APIs (OpenAI / Gemini)?
* *"Data Privacy (GDPR/HIPAA compliance prevents sending customer PII to 3rd-party clouds), Zero Recurring Token Cost, and Guaranteed Sub-Millisecond Latency with zero API downtime."*

---

## 🛡️ 4. The "Confidence Threshold Fallback" (Showstopper Feature)

**If asked:** *"What happens if a user submits a vague ticket like 'application is not working'?"*

> *"In real support operations, vague tickets lack specific category keywords. In our model, 'application is not working' produced a low confidence score of **33.7%**.*
> *Instead of blindly accepting the top guess and misrouting the user, I implemented a **Confidence Threshold Gate (default 40%)**.*
> *When confidence drops below 40%, the system activates an operational fallback:*
> 1. *Halts automated category dispatch.*
> 2. *Flags triage status as **'Manual Triage Required'**.*
> 3. *Sends an empathetic acknowledgment asking the customer for exact error codes or screenshots, while routing the ticket to the **Tier-1 Human Queue**."*

---

## 🥊 5. Tough Interview Questions & Bulletproof Answers

| Interviewer Question | Your Winning Answer |
| :--- | :--- |
| **"Why not use BERT or a Transformer?"** | *"On a dataset of 200 records, fine-tuning a 110M parameter BERT model leads to severe overfitting and requires GPU infrastructure. TF-IDF + Logistic Regression achieved 95% accuracy in 2ms on CPU without heavy dependencies. In a future iteration with 50,000+ tickets, I would deploy a lightweight local model like `all-MiniLM-L6-v2`."* |
| **"What is the difference between Macro F1 and Weighted F1?"** | *"Macro F1 calculates the metric independently for each category and averages them unweighted. Weighted F1 weights each score by category frequency. Because our classes are balanced, Macro F1 (94.79%) proves that performance is uniform across all 8 classes."* |
| **"Did you have duplicates in the data?"** | *"Full-row duplicates were 0. However, 80 unique descriptions appeared across 200 records, simulating real-world ticket frequency (different users reporting the same issue). I analyzed this in `src/preprocessing.py` and implemented deduplication toggles."* |
| **"How would you deploy this to production?"** | *"Dockerize the FastAPI application, deploy behind an NGINX reverse proxy on Kubernetes, log predictions with Prometheus/Grafana, and stream low-confidence edge cases to an active-learning pipeline for periodic retraining."* |

---

## 🗺️ 6. Codebase File Navigator

- **`src/preprocessing.py`**: Text cleaner (`clean_text`), schema validation, duplicate & null handling.
- **`src/eda.py`**: Generates distribution plots in `screenshots/`.
- **`src/train.py`**: Benchmarking 4 models via 5-Fold Stratified CV, training champion, evaluating confusion matrix, saving `models/ticket_classifier.joblib`.
- **`src/predict.py`**: Core inference engine, confidence probability calculation, confidence threshold fallback logic.
- **`src/responder.py`**: Offline customer response generator with SLAs and fallback clarification templates.
- **`src/app.py` & `static/`**: FastAPI REST API and modern glassmorphic web dashboard.
- **`tests/test_pipeline.py`**: 15 passing unit tests (`pytest`).
