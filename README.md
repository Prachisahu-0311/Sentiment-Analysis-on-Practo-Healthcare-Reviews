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

This section provides the details of the evaluation metrics for both **Baseline Models** and **Transformer-Based Models** on the real-world test set. The results highlight the improvements achieved by deep learning models for aspect-based sentiment analysis of healthcare reviews.

---

### **Baseline Models**
#### 1. **Logistic Regression (TF-IDF):**
- **Accuracy**: **95.21%**
- **Negative Class (Complaints):**
  - **Precision**: **40%**
  - **Recall**: **96%**
  - **F1-Score**: **56%**

#### 2. **XGBoost (TF-IDF):**
- **Accuracy**: **94.04%**
- **Negative Class (Complaints):**
  - **Precision**: **34%**
  - **Recall**: **93%**
  - **F1-Score**: **50%**

---

### **Transformer-Based Models**
#### 3. **ClinicalBERT (Fine-Tuned Model):**
- **Accuracy**: **97.91%**
- **Negative Class (Complaints):**
  - **Precision**: **61.59%**
  - **Recall**: **93.98%**
  - **F1-Score**: **74.41%**

#### 4. **RoBERTa (Twitter-RoBERTa Base):**
- **Accuracy**: **96.84%**
- **Negative Class (Complaints):**
  - **Precision**: **50.68%**
  - **Recall**: **96.21%**
  - **F1-Score**: **66.39%**

---

### **Key Takeaways**
1. **Baseline Models Established a Good Benchmark**:
   - Logistic Regression achieved high **accuracy (95.21%)** and **high negative recall (96%)**, proving it could detect most complaints.
   - However, **low negative precision (40%)** shows challenges with false positives due to the inability to understand context or negations.

2. **Deep Learning Outperformed Traditional Models**:
   - Both **ClinicalBERT** and **RoBERTa** significantly outperformed the baselines, improving **Negative F1-Scores** to **74.41%** and **66.39%**, respectively.
   - ClinicalBERT handled domain-specific medical jargon better than RoBERTa, resulting in substantially better **precision (61.59%)** for the negative class.

3. **Visualization and Dashboard**:
   - An **interactive dashboard** was built using Streamlit to compare model metrics and visualize results.
   - It includes features for real-time predictions using the fine-tuned transformer models.

---

### **Visualization Example**

#### 1. **Model Comparison Bar Chart**
![Model Comparison](path/to/your/bar_chart_image.png)

This bar chart displays the performance of all models on key metrics like Accuracy, Precision, Recall, and F1-Score.

#### 2. **Live Inference**
The dashboard includes a "Live Inference" tab where users can test custom healthcare reviews. The models analyze the review's sentiment and provide predictions in real-time.

**Example Input:**
```
"The doctor was nice but prescribed heavy antibiotics which gave my baby horrible side effects."
```

**ClinicalBERT Prediction:**
```
"Negative Sentiment"
```

---

These results and visualizations highlight the superiority of transformer-based models, like ClinicalBERT, for healthcare-related sentiment classification tasks while showcasing the improvements over traditional baselines.

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
