# Aspect-Based Sentiment Analysis on Healthcare Reviews

This repository contains the implementation of an **Aspect-Based Sentiment Analysis** (ABSA) pipeline for analyzing healthcare reviews. The project follows an end-to-end workflow, starting from data preprocessing, exploring baseline models using **TF-IDF**, fine-tuning **ClinicalBERT** and **RoBERTa**, and creating an interactive **Streamlit dashboard** for model evaluation and visualization.

---

## 📌 **Project Workflow**

### **1. Preprocessing**
- Cleaned and preprocessed raw healthcare review data.
- Applied:
  - Text tokenization and lemmatization.
  - Handling missing data and duplicates.
  - Removal of stopwords, special characters, and irrelevant tokens (like URLs).
  
---

### **2. Baseline Models**
- Implemented **TF-IDF** feature extraction to generate weighted term vectors.
- Trained baseline classifiers:
  - Logistic Regression.
  - Support Vector Machines (SVM).
  - Random Forest.
- Established baseline accuracy, precision, recall, and F1-scores for evaluation.

---

### **3. Fine-Tuned Models**
- Leveraged **Hugging Face Transformers** for transfer learning.
- Fine-tuned **ClinicalBERT** and **RoBERTa** pre-trained transformer models.
- Performed binary sentiment classification (Positive/Negative sentiment).
- Compared model performance using:
   - **Accuracy**
   - **Precision**
   - **Recall**
   - **F1-Score**

---

### **4. Dashboard Creation**
- Built an interactive **Streamlit dashboard** to:
  - Visualize model evaluation results.
  - Analyze predictions by ClinicalBERT and RoBERTa.
  - Compare metrics side-by-side.
  - Display key insights from classification results.

---

## 🛠️ **Tools and Technologies**

- **Python** - Data preprocessing, feature engineering, and modeling.
- **Hugging Face Transformers** - Using pre-trained transformer models (ClinicalBERT, RoBERTa).
- **PyTorch** - Fine-tuning transformer models.
- **TF-IDF** - For extracting weighted term features.
- **Scikit-learn** - For building baseline models and evaluating performance.
- **Pandas** and **NumPy** - For data loading, cleaning, and manipulation.
- **Matplotlib** and **Seaborn** - For visualization of metrics and data distributions.
- **Streamlit** - For deploying the results on an interactive dashboard.
- **Jupyter Notebooks** - For experimentation and result analysis.

---

## 🚀 **How to Run the Project**

### **1. Clone the Repository**
```bash
git clone https://github.com/<your-github-username>/<your-repository-name>.git
cd <your-repository-name>
```

### **2. Install Dependencies**
Create a virtual environment and install required Python packages:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

### **3. Run Preprocessing**
Preprocess the raw healthcare review data:
```bash
python preprocessing.py
```

---

### **4. Train Baseline Models**
Run TF-IDF feature extraction and train baseline models:
```bash
python baseline_models.py
```

---

### **5. Fine-Tune ClinicalBERT and RoBERTa**
Fine-tune the transformer models for binary classification tasks:
```bash
python fine_tune_clinical_roberta.py
```

---

### **6. Run the Streamlit Dashboard for Visualization**
Start the Streamlit app to explore model results and comparisons:
```bash
streamlit run app_hybrid_llm.py
```

Access the local dashboard at `http://localhost:8501/`.

---

## 📊 **Results**
- Baseline models achieved F1-scores between **[X%] and [Y%]**.
- Fine-tuned ClinicalBERT achieved an F1-score of **[Z.X%]**.
- Fine-tuned RoBERTa outperformed with an F1-score of **[W.X%]** (optional based on actual results).
- Dashboard visualizations provide actionable insights into model performance.

---

## 📌 **Highlights**
- Handled end-to-end **data preprocessing**, including cleaning, tokenization, and feature engineering.
- Built baseline **TF-IDF** models and fine-tuned **ClinicalBERT** and **RoBERTa**.
- Compared **transformer-based models** with traditional ML baselines using various evaluation metrics.
- Visualized results, comparisons, and key insights using an interactive **Streamlit Dashboard**.

---

## 🧰 **Repository Structure**

```
<project-name>/
|-- data/                             # Raw and processed datasets.
|-- llm_absa_models/                  # ABSA extractor and pipeline scripts.
    |-- mistral_7b_local/             # Directory for storing Mistral model files.
    |-- llm_absa_extractor.py         # Aspect-based sentiment model extractor.
    |-- llm_absa_run.py               # Full pipeline runner script.
    |-- app_hybrid_llm.py             # Streamlit dashboard app.
|-- transformer_models/               # Pretrained ClinicalBERT and RoBERTa fine-tuning code.
|-- absa_output/                      # Generated results from model evaluation.
|-- requirements.txt                  # Python dependencies.
|-- README.md                         # Project documentation (this file).
