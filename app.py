import streamlit as st
import fitz  # PyMuPDF
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page config
st.set_page_config(page_title="AI Resume Screener", layout="wide")

# 🔥 ANIMATED GRADIENT BACKGROUND
st.markdown("""
<style>

/* Animated Background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(-45deg, #0f2027, #203a43, #2c5364, #1c1c1c);
    background-size: 400% 400%;
    animation: gradientMove 12s ease infinite;
}

/* Animation */
@keyframes gradientMove {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Glass container */
[data-testid="stMain"] {
    background-color: rgba(0, 0, 0, 0.4);
    padding: 20px;
    border-radius: 15px;
}

/* Text */
h1, h2, h3, label {
    color: white !important;
}

/* Text area */
textarea {
    background-color: #1e1e1e !important;
    color: white !important;
    border-radius: 10px;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background-color: #1e1e1e;
    padding: 10px;
    border-radius: 10px;
}

/* Button */
.stButton > button {
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    border: none;
}

/* Progress bar */
.stProgress > div > div {
    background-color: #00c6ff;
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
    uploaded_files = st.file_uploader("📂 Upload Resumes (PDF)", accept_multiple_files=True)

# Extract text from PDF
def extract_text(file):
    text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

# Extract skills
def get_skills(text):
    words = text.lower().split()
    return set(words)

st.markdown("---")

# Button
if st.button("🚀 Screen Resumes"):

    if job_desc and uploaded_files:
        scores = []
        skill_results = []

        jd_skills = get_skills(job_desc)

        for file in uploaded_files:
            resume_text = extract_text(file)

            # Similarity calculation
            text_data = [job_desc, resume_text]
            cv = CountVectorizer().fit_transform(text_data)
            similarity = cosine_similarity(cv)[0][1]

            # Skill comparison
            resume_skills = get_skills(resume_text)
            matched = jd_skills.intersection(resume_skills)
            missing = jd_skills - resume_skills

            scores.append((file.name, round(similarity * 100, 2)))
            skill_results.append((file.name, matched, missing))

        # Sort results
        scores.sort(key=lambda x: x[1], reverse=True)

        df = pd.DataFrame(scores, columns=["Candidate", "Match %"])

        st.markdown("## 🏆 Results")

        # Top candidate
        top = df.iloc[0]
        st.success(f"🌟 Top Candidate: {top['Candidate']} ({top['Match %']}%)")

        # 📊 Pie chart (small)
        st.markdown("### 📊 Match Distribution")
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.pie(
            df["Match %"],
            labels=df["Candidate"],
            autopct='%1.1f%%',
            textprops={'fontsize': 8}
        )
        ax.set_title("Match Distribution")
        st.pyplot(fig)

        # Candidate details
        for i, (name, score) in enumerate(scores):
            st.markdown(f"### {name}")
            st.progress(int(score))

            matched, missing = skill_results[i][1], skill_results[i][2]

            st.markdown("**✅ Matched Skills:**")
            st.write(", ".join(list(matched)[:10]) if matched else "None")

            st.markdown("**❌ Missing Skills:**")
            st.write(", ".join(list(missing)[:10]) if missing else "None")

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
