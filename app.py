import streamlit as st
import fitz  # PyMuPDF
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page config
st.set_page_config(page_title="AI Resume Screener", layout="wide")

# CUSTOM CSS (🔥 MAIN DESIGN)
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}
.main {
    background-color: rgba(255, 255, 255, 0.05);
    padding: 20px;
    border-radius: 15px;
}
h1, h2, h3 {
    text-align: center;
    color: white;
}
.stTextArea textarea {
    background-color: #1e1e1e;
    color: white;
}
.stFileUploader {
    background-color: #1e1e1e;
    padding: 10px;
    border-radius: 10px;
}
.stButton>button {
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1>🚀 AI Resume Screening Tool</h1>", unsafe_allow_html=True)
st.markdown("<h3>Smart Shortlisting for Recruiters</h3>", unsafe_allow_html=True)

st.markdown("---")

# Layout
col1, col2 = st.columns(2)

with col1:
    job_desc = st.text_area("📄 Job Description", height=250)

with col2:
    uploaded_files = st.file_uploader("📂 Upload Resumes", accept_multiple_files=True)

# Extract text
def extract_text(file):
    text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

st.markdown("---")

# Button
if st.button("🚀 Screen Resumes"):

    if job_desc and uploaded_files:
        scores = []

        for file in uploaded_files:
            resume_text = extract_text(file)

            text_data = [job_desc, resume_text]
            cv = CountVectorizer().fit_transform(text_data)
            similarity = cosine_similarity(cv)[0][1]

            scores.append((file.name, round(similarity * 100, 2)))

        scores.sort(key=lambda x: x[1], reverse=True)

        df = pd.DataFrame(scores, columns=["Candidate", "Match %"])

        st.markdown("## 🏆 Results")

        # Top candidate
        top = df.iloc[0]
        st.success(f"🌟 Top Candidate: {top['Candidate']} ({top['Match %']}%)")

        for name, score in scores:
            st.markdown(f"### {name}")
            st.progress(int(score))

            if score >= 70:
                st.success("Excellent Match ✅")
            elif score >= 50:
                st.warning("Good Match ⚠️")
            else:
                st.error("Low Match ❌")

            st.markdown("---")

        # Download
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Results", csv, "results.csv", "text/csv")

    else:
        st.warning("⚠️ Please enter job description and upload resumes")
