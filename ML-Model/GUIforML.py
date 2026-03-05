import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# 1. PATH CONFIGURATION
base_dir = os.path.dirname(__file__)
model_path = os.path.join(base_dir, 'os_lncrna_detector.pkl')

# 2. LOAD THE MODEL
@st.cache_resource
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

# 4. SIDEBAR - FIXED INSTRUCTION MESSAGE
st.sidebar.header("User Control Panel")

# This message will always show in the sidebar
st.sidebar.markdown("""
**Instructions:**
1. Click 'Browse files' below.
2. Upload a **.tsv** file from your GDC batch.
3. The AI will extract **MALAT1** and **NEAT1** to assess risk.
""")

uploaded_file = st.sidebar.file_uploader("Upload Genomic Profile (.tsv)", type="tsv")

# 5. MAIN LOGIC
if model is None:
    st.error("Critical Error: 'os_lncrna_detector.pkl' not found. Please ensure the model file is in the same folder as this script.")

elif uploaded_file is not None:
    try:
        # Standardize data reading
        df = pd.read_csv(uploaded_file, sep='\t')
        df.columns = [c.lower() for c in df.columns]

        # Check if we need to skip headers (STAR counts specific)
        if 'gene_name' not in df.columns and 'gene symbol' not in df.columns:
            df = pd.read_csv(uploaded_file, sep='\t', skiprows=1)
            df.columns = [c.lower() for c in df.columns]

        # Identify Column Names
        gene_col = 'gene_name' if 'gene_name' in df.columns else 'gene symbol'
        val_col = 'tpm_unstranded' if 'tpm_unstranded' in df.columns else 'copy_number' if 'copy_number' in df.columns else df.columns[1]

        # Extract Values
        m1_data = df[df[gene_col].str.contains('MALAT1', na=False, case=False)]
        n1_data = df[df[gene_col].str.contains('NEAT1', na=False, case=False)]
        
        if not m1_data.empty and not n1_data.empty:
            malat1 = m1_data[val_col].values[0]
            neat1 = n1_data[val_col].values[0]
            
            st.write(f"### Analysis for {uploaded_file.name}")
            col1, col2 = st.columns(2)
            col1.metric("MALAT1 Level", f"{malat1:.2f}")
            col2.metric("NEAT1 Level", f"{neat1:.2f}")

            # Predict
            sample = pd.DataFrame([[malat1, neat1]], columns=['MALAT1', 'NEAT1'])
            prob = model.predict_proba(sample)[0][1]
            
            st.write("---")
            st.write("## AI Diagnostic Result")
            
            if prob > 0.75:
                st.error(f"🚨 **High Risk Detected**: {prob*100:.1f}% probability")
            elif prob > 0.4:
                st.warning(f"⚠️ **Moderate Risk / Borderline**: {prob*100:.1f}% probability")
            else:
                st.success(f"✅ **Normal/Healthy Signature**: {prob*100:.1f}% probability")
            
            st.progress(prob)
            
        else:
            st.warning("⚠️ MALAT1 or NEAT1 not found. Ensure your file uses standard GDC gene symbols.")

    except Exception as e:
        st.error(f"Error processing file: {e}")

else:
    # This shows in the MAIN window when no file is uploaded
    st.info("Please upload a .tsv file in the sidebar to begin the AI analysis.")

# 6. FOOTER
st.markdown("---")
st.caption("MIRACLE v1.3 | Powered by Multi-Omic Random Forest | ASU 2026")
