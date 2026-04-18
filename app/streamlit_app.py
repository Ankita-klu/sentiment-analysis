import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
# Set the page to wide mode to better accommodate dataframes and plots
st.set_page_config(page_title="Sentiment Analysis Evaluation", layout="wide")

st.title("Twitter Entity Sentiment Analysis - Evaluation Dashboard")
st.write("This dashboard presents the evaluation metrics for the 4-class sentiment classification model.")
st.markdown("---")

# ---------------------------------------------------------
# 1. Data Loading / Simulation Module
# ---------------------------------------------------------
@st.cache_data
def generate_dummy_data():
    """
    Generates simulated true labels and predicted labels to facilitate 
    dashboard development before the final ML model is integrated.
    The dummy data mimics a 4-class classification problem.
    """
    categories = ['Negative', 'Neutral', 'Positive', 'Irrelevant']
    n_samples = 300
    
    # Simulate a realistic distribution of true labels in the dataset
    np.random.seed(42) # Ensuring reproducible results across app reloads
    y_true = np.random.choice(categories, size=n_samples, p=[0.25, 0.35, 0.30, 0.10])
    
    # Simulate model predictions with a programmed error margin (approx. 80% accuracy)
    y_pred = []
    dummy_tweets = []
    
    for i, true_label in enumerate(y_true):
        dummy_tweets.append(f"Simulated tweet content #{i} regarding a specific entity.")
        
        if np.random.rand() > 0.20: 
            y_pred.append(true_label)
        else:
            # Randomly select an incorrect category if a simulated error occurs
            wrong_categories = [c for c in categories if c != true_label]
            y_pred.append(np.random.choice(wrong_categories))
            
    # Structure the simulated data into a pandas DataFrame
    df = pd.DataFrame({
        'Tweet': dummy_tweets,
        'True_Label': y_true,
        'Predicted_Label': y_pred
    })
    
    return df, categories

# Execution of data generation
data_df, class_labels = generate_dummy_data()

# Sidebar notification regarding development status
st.sidebar.header("System Status")
st.sidebar.warning("DEVELOPMENT MODE: Utilizing simulated dummy data. This module will be connected to actual CSV/Model outputs upon completion by the Modeling team.")

# ---------------------------------------------------------
# 2. Evaluation Metrics Module
# ---------------------------------------------------------
st.header("1. Core Performance Metrics")
st.write("Quantitative analysis of model performance focusing on the Macro F1 Score.")

# Calculate Macro F1 Score: harmonic mean of precision and recall
# Crucial for assessing performance on imbalanced classification tasks
macro_f1 = f1_score(data_df['True_Label'], data_df['Predicted_Label'], average='macro')

col1, col2 = st.columns([1, 2])

with col1:
    st.metric(label="Overall F1 Score (Macro)", value=f"{macro_f1:.4f}")
    st.caption("Target: Close to 1.0. Measures balance between Precision and Recall.")

with col2:
    st.write("**Detailed Classification Report**")
    # Generate report as dictionary to transform into a styled DataFrame
    report_dict = classification_report(data_df['True_Label'], data_df['Predicted_Label'], output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    # Note: Using width='stretch' to adhere to updated Streamlit API standards
    st.dataframe(report_df.style.format("{:.2f}"), width='stretch')

st.markdown("---")

# ---------------------------------------------------------
# 3. Confusion Matrix Module
# ---------------------------------------------------------
st.header("2. Confusion Matrix Heatmap")
st.write("Visualizes classification overlaps. Diagonal axis indicates correct predictions.")

# Computation of the confusion matrix array
cm = confusion_matrix(data_df['True_Label'], data_df['Predicted_Label'], labels=class_labels)

# Matplotlib/Seaborn rendering pipeline
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_labels, yticklabels=class_labels, ax=ax)
ax.set_ylabel('Actual Category (Ground Truth)')
ax.set_xlabel('Predicted Category (Model Output)')
ax.set_title('Confusion Matrix Visualization')

st.pyplot(fig)

st.markdown("---")

# ---------------------------------------------------------
# 4. Advanced Qualitative Error Analysis Module
# ---------------------------------------------------------
st.header("3. Diagnostic Error Analysis")
st.write("This tool categorizes misclassifications to provide actionable feedback for the Data and Modeling teams.")

# Isolate misclassified instances for detailed review
misclassified_df = data_df[data_df['True_Label'] != data_df['Predicted_Label']].copy()

# Simulate typical Twitter NLP error typologies for demonstration purposes
error_archetypes = ['Sarcasm / Irony', 'Entity Confusion', 'Slang / Emoji Dropped', 'Double Negation']
np.random.seed(10)
misclassified_df['Probable_Error_Cause'] = np.random.choice(error_archetypes, size=len(misclassified_df))

st.subheader("Error Distribution")
# Aggregate and visualize the distribution of error typologies
error_counts = misclassified_df['Probable_Error_Cause'].value_counts()
st.bar_chart(error_counts)

# Interactive workbench for targeted error investigation
st.subheader("Interactive Investigation Workbench")
selected_error_type = st.selectbox("Filter errors by suspected NLP challenge:", ['All'] + error_archetypes)

# Apply dynamic filtering based on user selection
if selected_error_type == 'All':
    filtered_df = misclassified_df
else:
    filtered_df = misclassified_df[misclassified_df['Probable_Error_Cause'] == selected_error_type]

st.dataframe(filtered_df[['Tweet', 'True_Label', 'Predicted_Label', 'Probable_Error_Cause']], width='stretch')

# Dynamically generate architectural and preprocessing recommendations
st.markdown("### Actionable Insights for the Team")
if selected_error_type == 'Sarcasm / Irony':
    st.warning("**To Modeling Team:** TF-IDF fails to capture the contrasting context of sarcasm. Consider transitioning to a transformer-based model (e.g., DistilBERT) to capture bidirectional context.")
elif selected_error_type == 'Slang / Emoji Dropped':
    st.warning("**To Data Team (Ankita):** Traditional regex cleaning might be erasing critical sentiment markers (emojis). Implement an emoji-to-text mapping step before tokenization.")
elif selected_error_type == 'Entity Confusion':
    st.warning("**To Modeling Team:** The model evaluates sentence-level sentiment instead of entity-level. Suggest extracting a text window (e.g., 5 words surrounding the target entity) for training.")
elif selected_error_type == 'Double Negation':
    st.warning("**To Modeling Team:** 'Not bad' is being tokenized into separate negative vectors. Increase the n-gram range in the vectorizer to include bi-grams (ngram_range=(1, 2)).")
else:
    st.info("Select a specific error type above to generate tailored architectural recommendations.")