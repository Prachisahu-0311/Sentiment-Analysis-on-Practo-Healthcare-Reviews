import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="Practo Insights Dashboard", layout="wide")
st.title("🏥 Healthcare Review Insights & Sentiment Analysis")
st.markdown("Baseline Model: **TF-IDF + Logistic Regression**")

# ==========================================
# PATHS (Updated for your SSH Linux environment)
# ==========================================
# Find the absolute path based on where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR) # Go up one level to my_project_m24mac005
BASE_DIR = os.path.join(ROOT_DIR, "Doctor_reviews_scraped_text_data", "Practo_scraped_text_data", "output_processed_data")

MASTER_DATA_PATH = os.path.join(BASE_DIR, "dashboard_master_data.csv")
MODEL_DIR = os.path.join(SCRIPT_DIR, "saved_models")

# ==========================================
# CACHED DATA & MODEL LOADING
# ==========================================
@st.cache_data
def load_data():
    df = pd.read_csv(MASTER_DATA_PATH)
    return df

@st.cache_resource
def load_models():
    vectorizer = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
    model = joblib.load(os.path.join(MODEL_DIR, "baseline_lr_model.pkl"))
    return vectorizer, model

with st.spinner("Loading 290k+ Real-World Reviews..."):
    df = load_data()
    vectorizer, model = load_models()

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
st.sidebar.header("Navigation")
menu = st.sidebar.radio("Go to:", ["📊 Dataset Overview", "👨‍⚕️ Doctor Analysis", "🧠 Real-Time Sentiment Prediction"])

# ==========================================
# TAB 1: DATASET OVERVIEW
# ==========================================
if menu == "📊 Dataset Overview":
    st.header("Real-World Dataset Statistics")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Reviews", f"{len(df):,}")
    col2.metric("Total Doctors", f"{df['doctor_name'].nunique():,}")
    col3.metric("Total Cities", f"{df['city'].nunique():,}")
    
    st.markdown("---")
    colA, colB = st.columns(2)
    
    with colA:
        st.subheader("Sentiment Distribution")
        sentiment_counts = df['sentiment_label'].value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Count']
        fig_pie = px.pie(sentiment_counts, values='Count', names='Sentiment', color='Sentiment',
                         color_discrete_map={'positive':'#2ecc71', 'negative':'#e74c3c'})
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with colB:
        st.subheader("Top Specializations")
        spec_counts = df['specialization'].value_counts().head(10).reset_index()
        spec_counts.columns = ['Specialization', 'Review Count']
        fig_bar = px.bar(spec_counts, x='Review Count', y='Specialization', orientation='h')
        fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_bar, use_container_width=True)

# ==========================================
# TAB 2: DOCTOR ANALYSIS
# ==========================================
elif menu == "👨‍⚕️ Doctor Analysis":
    st.header("Doctor-Level Insights")
    
    col1, col2, col3 = st.columns(3)
    selected_city = col1.selectbox("Select City", df['city'].dropna().unique())
    filtered_city_df = df[df['city'] == selected_city]
    
    selected_spec = col2.selectbox("Select Specialization", filtered_city_df['specialization'].dropna().unique())
    filtered_spec_df = filtered_city_df[filtered_city_df['specialization'] == selected_spec]
    
    selected_doctor = col3.selectbox("Select Doctor", filtered_spec_df['doctor_name'].dropna().unique())
    doc_df = filtered_spec_df[filtered_spec_df['doctor_name'] == selected_doctor]
    
    st.markdown(f"### Reviews for {selected_doctor}")
    st.write(f"**Practice:** {doc_df['practice_name'].iloc[0] if not doc_df['practice_name'].empty else 'N/A'}")
    
    total_doc_reviews = len(doc_df)
    positive_reviews = len(doc_df[doc_df['sentiment_label'] == 'positive'])
    pos_percentage = (positive_reviews / total_doc_reviews) * 100 if total_doc_reviews > 0 else 0
    
    colA, colB = st.columns(2)
    colA.metric("Total Reviews", total_doc_reviews)
    colB.metric("Patient Recommendation Rate", f"{pos_percentage:.1f}%")
    
    st.markdown("#### Patient Contexts & Tags Mentioned")
    tags_mentioned = doc_df['tags'].dropna().str.split(',').explode().str.strip()
    tags_mentioned = tags_mentioned[tags_mentioned != ''].value_counts().head(5)
    if not tags_mentioned.empty:
        st.bar_chart(tags_mentioned)
    else:
        st.info("No specific tags recorded for this doctor.")
        
    st.markdown("#### Recent Reviews")
    st.dataframe(doc_df[['reviewed_on', 'sentiment_label', 'review_text_clean', 'reply_text']].head(10), use_container_width=True)

# ==========================================
# TAB 3: REAL-TIME INFERENCE
# ==========================================
elif menu == "🧠 Real-Time Sentiment Prediction":
    st.header("Test the Baseline Model")
    st.markdown("Paste a healthcare review below. The TF-IDF + Logistic Regression model will predict its sentiment instantly.")
    
    user_input = st.text_area("Enter review text:", height=150, placeholder="E.g., The doctor was very nice, but I had to wait 2 hours in the lobby...")
    
    if st.button("Predict Sentiment"):
        if user_input.strip() == "":
            st.warning("Please enter some text.")
        else:
            vec_input = vectorizer.transform([user_input])
            prediction = model.predict(vec_input)[0]
            probabilities = model.predict_proba(vec_input)[0]
            
            if prediction == 'positive':
                st.success(f"**Prediction:** POSITIVE (Confidence: {probabilities[1]*100:.2f}%)")
            else:
                st.error(f"**Prediction:** NEGATIVE (Confidence: {probabilities[0]*100:.2f}%)")