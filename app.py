import streamlit as st
import fitz  # PyMuPDF
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page Config
st.set_page_config(page_title="AI Resume Screener", layout="wide")

# Title
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>AI Resume Screening Tool</h1>", unsafe_allow_html=True)
st.markdown("---")

# Layout
col1, col2 = st.columns(2)

# Job Description
with col1:
    st.subheader("📄 Job Description")
    job_desc = st.text_area("Paste Job Description Here", height=250)

# Resume Upload
with col2:
    st.subheader("📂 Upload Resumes")
    uploaded_files = st.file_uploader("Upload PDF files", accept_multiple_files=True)

# Extract text from PDF
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

        st.subheader("🏆 Candidate Ranking")

        for name, score in scores:
            if score >= 70:
                st.success(f"✅ {name} — {score}% (Top Candidate)")
            elif score >= 50:
                st.warning(f"⚠️ {name} — {score}% (Good Match)")
            else:
                st.error(f"❌ {name} — {score}% (Low Match)")

    else:
        st.warning("⚠️ Please upload resumes and enter job description")
