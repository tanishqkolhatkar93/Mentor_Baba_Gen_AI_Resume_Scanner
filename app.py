# streamlit_demo.py
import streamlit as st
import joblib
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# Define the TextCleaner class (copy from your notebook)
class TextCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, remove_stopwords=False):
        self.remove_stopwords = remove_stopwords
        if remove_stopwords:
            # very small stop list for demo (you can use nltk or sklearn stop words)
            self.stopwords = set(["the","and","a","an","in","on","with","of","for","to","is","are","some"])
        else:
            self.stopwords = set()

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)  # remove punctuation (keep alphanumerics)
        tokens = text.split()
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.stopwords]
        return " ".join(tokens)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [self.clean_text(t) for t in X]

# Create a small dummy dataset (copy from your notebook)
data = [
    # (JD, Resume, Label)
    ("Looking for a Python developer with experience in ML and AWS.",
     "Experienced Python engineer skilled in machine learning, cloud computing and AWS services.",
     1),

    ("Senior frontend role: React, TypeScript, HTML/CSS required.",
     "Frontend developer with experience in React, TypeScript, HTML and CSS.",
     1),

    ("Need a data analyst proficient in SQL and Excel. Some Python a plus.",
     "Marketing manager with strong Excel skills but no SQL experience.",
     0),

    ("DevOps engineer: Docker, Kubernetes, CI/CD pipelines.",
     "Worked on Docker containers and Kubernetes clusters and CI/CD automation.",
     1),

    ("Looking for a Java backend developer (Spring Boot).",
     "Java developer experienced in Spring Boot, microservices and REST APIs.",
     1),

    ("Mobile developer for Android (Kotlin) required.",
     "iOS developer experienced in Swift and Objective-C, no Android experience.",
     0),

    ("Entry-level role: good communication, basic Python knowledge ok.",
     "Recent graduate with excellent communication and some Python coursework.",
     1),

    ("Hiring data scientist: deep learning, PyTorch or TensorFlow.",
     "Applied deep learning projects using PyTorch and TensorFlow for image tasks.",
     1),

    ("Sales person required with experience in B2B software sales.",
     "Customer support person, experience in SaaS customer success (not sales).",
     0),

    ("Full-stack position: Node.js, Express, React and MongoDB.",
     "Full-stack engineer: Node.js, Express, React, MongoDB and REST APIs.",
     1),

    ("Security analyst: knowledge of networking and intrusion detection.",
     "Network engineer with experience in routers and switches, limited IDS exposure.",
     0),

    ("Cloud engineer: Azure experience and infrastructure as code (Terraform).",
     "Cloud engineer experienced with Azure and Terraform deployments.",
     1),
]

df = pd.DataFrame(data, columns=["jd", "resume", "label"])
df["combined"] = df["jd"] + " [SEP] " + df["resume"]
X = df["combined"].values
y = df["label"].values

# Build and train the pipeline (copy from your notebook)
pipe = Pipeline([
    ("cleaner", TextCleaner(remove_stopwords=False)),
    ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=2000)),
    ("clf", LogisticRegression(max_iter=200, solver="liblinear"))
])

pipe.fit(X, y) # Train on the full dataset for the demo

# Save the pipeline to disk (moved after pipe is defined and trained)
joblib.dump(pipe, "resume_screener_pipe.joblib")

# Load the pipeline from disk (this will now work after saving)
# pipe = joblib.load("resume_screener_pipe.joblib") # No need to load again, it's already in memory


st.title("MentorBaba — Resume Screener (Demo)")
jd = st.text_area("Paste Job Description", height=120)
resume = st.text_area("Paste Resume text", height=200)

if st.button("Evaluate Match"):
    combined = jd + " [SEP] " + resume
    # The pipeline expects a list of strings, even for a single input
    prob = pipe.predict_proba([combined])[0,1]
    st.write(f"Match probability: **{prob:.2%}**")
    st.success("MATCH" if prob > 0.5 else "NO MATCH")