import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, 
    accuracy_score, roc_curve, auc
)
from sklearn.preprocessing import LabelBinarizer

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
# Configuring the Streamlit page settings for optimal visualization
st.set_page_config(page_title="Twitter Sentiment Analysis Evaluation", layout="wide")

st.title("Twitter Entity Sentiment Analysis - Final Evaluation")
st.write("This dashboard presents the final performance metrics derived from the actual trained model and validation dataset.")
st.markdown("---")

# ---------------------------------------------------------
# 0. Exploratory Data Analysis (Static Overviews)
# ---------------------------------------------------------
# Displaying pre-generated EDA visualizations from the training phase
st.header("0. Exploratory Data Analysis (EDA)")
st.write("Overview of the training dataset characteristics provided by the Data & Modeling team.")

# First row of EDA visualizations
eda_col1, eda_col2 = st.columns(2)

with eda_col1:
    st.subheader("Class Distribution")
    try:
        st.image("data/class_distribution.png", caption="Sentiment class balance in the dataset", use_container_width=True)
    except Exception:
        st.warning("File 'class_distribution.png' not found in the data directory.")

with eda_col2:
    st.subheader("Word Cloud")
    try:
        st.image("data/wordclouds.png", caption="Most frequent words across sentiments", use_container_width=True)
    except Exception:
        st.warning("File 'wordclouds.png' not found in the data directory.")

# Second row of EDA visualizations
eda_col3, eda_col4 = st.columns(2)

with eda_col3:
    st.subheader("Tweet Length Distribution")
    try:
        st.image("data/tweet_length_distribution.png", caption="Distribution of tweet lengths by sentiment", use_container_width=True)
    except Exception:
        st.warning("File 'tweet_length_distribution.png' not found.")

with eda_col4:
    st.subheader("Top Topics and Entities")
    try:
        st.image("data/top_topics.png", caption="Most frequently discussed entities in the dataset", use_container_width=True)
    except Exception:
        st.warning("File 'top_topics.png' not found.")

st.markdown("---")

# ---------------------------------------------------------
# 1. Core Model and Data Loading
# ---------------------------------------------------------
@st.cache_resource
def load_final_assets():
    """
    Loads the trained model and vectorizer using joblib.
    Retrieves the validation dataset for performance assessment.
    """
    try:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        data_dir = os.path.join(repo_root, 'data')

        # Loading model components from the centralized data directory
        model = joblib.load(os.path.join(data_dir, 'best_model.pkl'))
        vectorizer = joblib.load(os.path.join(data_dir, 'vectorizer.pkl'))

        # Loading the validation dataset from the preprocessed data directory
        val_df = pd.read_csv(os.path.join(data_dir, 'val_clean.csv'))
        val_df = val_df.dropna(subset=['clean_tweet'])
        val_df.rename(columns={
            'tweet_id': 'ID',
            'topic': 'Entity',
            'sentiment': 'True_Label',
            'tweet': 'Tweet'
        }, inplace=True)

        return model, vectorizer, val_df
    except Exception as e:
        st.error(f"Critical Error loading project assets: {e}")
        return None, None, None

model, vectorizer, val_df = load_final_assets()

if model and vectorizer and val_df is not None:
    # ---------------------------------------------------------
    # 2. Real-time Evaluation Pipeline
    # ---------------------------------------------------------
    # Processing the validation set through the inference pipeline
    with st.spinner('Executing model inference on the validation set...'):
        X_val = vectorizer.transform(val_df['clean_tweet'])
        y_true = val_df['True_Label']
        y_pred = model.predict(X_val)
        
        # Handling LinearSVC decision function vs probabilistic outputs
        if hasattr(model, "predict_proba"):
            y_scores = model.predict_proba(X_val)
        else:
            y_scores = model.decision_function(X_val)
            
        class_labels = sorted(y_true.unique())
        val_df['Predicted_Label'] = y_pred

    # ---------------------------------------------------------
    # 3. Section 1: Core Performance Metrics
    # ---------------------------------------------------------
    st.header("1. Core Performance Metrics")
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')

    m_col1, m_col2, m_col3 = st.columns([1, 1, 2])
    with m_col1:
        st.metric("Top-1 Accuracy", f"{acc*100:.2f}%")
    with m_col2:
        st.metric("Macro F1 Score", f"{f1:.4f}")
    with m_col3:
        st.write("**Classification Report Summary**")
        report = classification_report(y_true, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose().style.format("{:.2f}"), width='stretch')

    # Displaying the historical model comparison results
    st.markdown("### Model Selection (Training Phase)")
    st.write("Performance benchmarks for different algorithms tested during the development phase.")
    try:
        st.image("data/model_comparison.png", caption="Comparative analysis of F1 Scores", width=600)
    except Exception:
        st.info("Model comparison visualization not available.")

    st.markdown("---")

    # ---------------------------------------------------------
    # 4. Section 2 & 3: Model Diagnostics (Advanced Layout)
    # ---------------------------------------------------------
    st.header("2. Model Diagnostics")
    st.write("Visual evaluation of model discrimination power and prediction overlaps.")
    
    # Implementing a side-by-side layout for compact visualization
    diag_col1, diag_col2 = st.columns(2)
    
    # Left column: ROC-AUC analysis
    with diag_col1:
        st.subheader("ROC-AUC Analysis")
        lb = LabelBinarizer()
        y_true_bin = lb.fit_transform(y_true)
        
        # Reduced figure size for optimal column fitting
        fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
        for i, label in enumerate(lb.classes_):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
            roc_auc = auc(fpr, tpr)
            ax_roc.plot(fpr, tpr, label=f'Class {label} (AUC = {roc_auc:.2f})')
        
        ax_roc.plot([0, 1], [0, 1], color='navy', linestyle='--')
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.legend(loc='lower right')
        st.pyplot(fig_roc)

    # Right column: Confusion matrix
    with diag_col2:
        st.subheader("Confusion Matrix Heatmap")
        cm = confusion_matrix(y_true, y_pred, labels=class_labels)
        
        # Maintaining consistent height with the ROC curve
        fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_labels, yticklabels=class_labels, ax=ax_cm)
        ax_cm.set_ylabel('Actual Category')
        ax_cm.set_xlabel('Predicted Category')
        st.pyplot(fig_cm)

    st.markdown("---")

    # ---------------------------------------------------------
    # 5. Section 4: Qualitative Error Analysis
    # ---------------------------------------------------------
    st.header("3. Qualitative Error Analysis")
    errors = val_df[val_df['True_Label'] != val_df['Predicted_Label']].copy()
    
    # Dictionary mapping error types to their NLP definitions
    error_definitions = {
        'Sarcasm': 'Instances where ironic context was misinterpreted.',
        'Negation': 'Complex negative structures resulting in sentiment inversion.',
        'Short Text': 'Brief tweets providing insufficient semantic context.',
        'Entity Confusion': 'Sentiment incorrectly associated with the target entity.'
    }
    
    st.write(f"Identified {len(errors)} misclassified samples. Review detailed logs below:")
    st.dataframe(errors[['Entity', 'Tweet', 'True_Label', 'Predicted_Label']].head(10), width='stretch')
    
    st.markdown("### Technical Recommendations")
    selected_pattern = st.selectbox("Select an observed error pattern for optimization strategies:", list(error_definitions.keys()))
    st.info(f"Root Cause Analysis: {error_definitions[selected_pattern]}")
    
    # Generating dynamic recommendations based on error typology
    if selected_pattern == 'Sarcasm':
        st.warning("Actionable Insight: Transition to Transformer-based architectures (e.g., DistilBERT) to capture long-range bidirectional dependencies.")
    elif selected_pattern == 'Negation':
        st.warning("Actionable Insight: Implement higher-order n-grams (1, 3) to preserve local semantic negation structures.")
    else:
        st.success("Actionable Insight: Enhance the preprocessing pipeline to preserve emojis and domain-specific slang as critical sentiment indicators.")

    st.markdown("---")

    # ---------------------------------------------------------
    # 6. Section 5: Interactive Live Predictor
    # ---------------------------------------------------------
    st.header("4. Interactive Live Predictor")
    st.write("Deployment of the production model for real-time inference on arbitrary text inputs.")
    
    input_text = st.text_area("Input text for sentiment analysis:", placeholder="Type a tweet or review here...")
    
    if st.button("Evaluate Sentiment", type="primary"):
        if input_text:
            vec_input = vectorizer.transform([input_text])
            live_pred = model.predict(vec_input)[0]
            
            # Supporting both probabilistic and decision-based classification metrics
            if hasattr(model, "predict_proba"):
                live_prob = max(model.predict_proba(vec_input)[0]) * 100
                st.success(f"Predicted Sentiment: {live_pred} (Confidence: {live_prob:.2f}%)")
            else:
                live_score = max(model.decision_function(vec_input)[0])
                st.success(f"Predicted Sentiment: {live_pred} (Decision Confidence Score: {live_score:.2f})")

else:
    st.error("System assets not found. Verify the presence of 'best_model.pkl' and 'vectorizer.pkl' in the data directory.")