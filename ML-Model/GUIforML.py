import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# 1. PATH CONFIGURATION
# Using __file__ ensures it works on Streamlit Cloud regardless of folder structure
base_dir = os.path.dirname(__file__)
model_path = os.path.join(base_dir, 'os_lncrna_detector.pkl')

# 2. LOAD THE MODEL
@st.cache_resource # This keeps the app fast by loading the model only once
def load_model():
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_model()

# 3. GUI BRANDING
st.set_page_config(page_title="MIRACLE Project | ASU", page_icon="🧬")

st.title("MIRACLE Project: Osteosarcoma lncRNA Detector")
st.markdown("### Albany State University Undergraduate Research 2025-2026 Cohort")
st.markdown("**Developed by:** Ayodeji Williams | Albany State University Bioinformatics")
st.markdown("**Mentors:** Dr. Olabisi Ojo & Dr. Wanjun Hu")
st.markdown("*College of Arts and Science | Dept. of Math, CS and Physics & Dept. of Natural Sciences*")

st.write("---")

# 4. SIDEBAR INPUT
st.sidebar.header("User Input")
uploaded_file = st.sidebar.file_uploader("Upload Genomic Profile (.tsv)", type="tsv")

if model is None:
    st.error("Critial Error: 'os_lncrna_detector.pkl' not found in the repository.")
elif uploaded_file is not None:
    try:
        # Read the dataset
        # GDC STAR counts usually need skiprows=1, but CNV/Methylation don't.
        # We'll read the file and then check for headers.
        df = pd.read_csv(uploaded_file, sep='\t')
        
        # Standardize columns to lowercase to prevent naming errors
        df.columns = [c.lower() for c in df.columns]

        # Determine if we need to skip the first row (common in GDC STAR files)
        if 'gene_name' not in df.columns and 'gene symbol' not in df.columns:
            # Re-read with skip if the first attempt didn't find headers
            df = pd.read_csv(uploaded_file, sep='\t', skiprows=1)
            df.columns = [c.lower() for c in df.columns]

        # Identify Column Names automatically
        gene_col = 'gene_name' if 'gene_name' in df.columns else 'gene symbol'
        val_col = 'tpm_unstranded' if 'tpm_unstranded' in df.columns else 'copy_number' if 'copy_number' in df.columns else df.columns[1]

        # Extract the necessary lncRNA values
        m1_data = df[df[gene_col].str.contains('MALAT1', na=False, case=False)]
        n1_data = df[df[gene_col].str.contains('NEAT1', na=False, case=False)]
        
        if not m1_data.empty and not n1_data.empty:
            malat1 = m1_data[val_col].values[0]
            neat1 = n1_data[val_col].values[0]
            
            # Display extracted values
            st.write(f"### Analysis for {uploaded_file.name}")
            col1, col2 = st.columns(2)
            col1.metric("MALAT1 Level", f"{malat1:.2f}")
            col2.metric("NEAT1 Level", f"{neat1:.2f}")

            # Perform Prediction
            sample = pd.DataFrame([[malat1, neat1]], columns=['MALAT1', 'NEAT1'])
            prob = model.predict_proba(sample)[0][1]
            
            st.write("---")
            st.write("## Detection Result")
            
            # Diagnostic Logic (Fixed indentation here)
            if prob > 0.75:
                st.error(f"High Risk of Osteosarcoma Signature: {prob*100:.1f}% probability")
            elif prob > 0.4:
                st.warning(f"Moderate Risk / Borderline Signature: {prob*100:.1f}% probability")
            else:
                st.success(f"Normal/Healthy Signature: {prob*100:.1f}% probability")
            
            # Visual Progress Bar
            st.progress(prob)
            
        else:
            st.error("Markers MALAT1 or NEAT1 not found. Ensure the file contains these gene names.")

    except Exception as e:
        st.error(f"Error processing file: {e}")

# 5. FOOTER
st.markdown("---")
st.caption("MIRACLE v1.3 | Trained on GDC Multi-Omic Data | 2026 ASU Cohort")
