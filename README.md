# 📄 Smart Resume Screener 

🚀 **Objective:**  
This project builds a **machine learning model** that predicts whether a resume matches a given job description (JD).  
It automates resume screening using **text preprocessing + feature extraction + classification**, helping recruiters save time while ensuring better candidate-job alignment.  

---

## ✨ Features
- ✅ **Dataset Creation** — dummy dataset of JD/Resume pairs with labels (Match = 1, No Match = 0)  
- ✅ **Preprocessing** — lowercasing, punctuation removal, optional stopwords removal  
- ✅ **Feature Engineering** — TF-IDF vectorization (unigrams + bigrams)  
- ✅ **Model Training** — Logistic Regression classifier  
- ✅ **Evaluation** — Accuracy, Precision, Recall, F1-Score, Confusion Matrix  
- ✅ **Predictions** — Test on new JD/Resume pairs  
- ✅ **(Optional)** Quick Streamlit demo for uploading JD & Resume and getting match probability  

---

## 🛠️ Tech Stack
- **Python 3.10+**  
- **Libraries**:  
  - `pandas`, `numpy` → data handling  
  - `scikit-learn` → preprocessing, TF-IDF, ML models  
  - `matplotlib`, `seaborn` → visualization (optional)  
  - `streamlit` → dashboard demo (optional)  

---

## 📂 Project Structure
```
📦 smart-resume-screener
 ┣ 📜 resume_screener.ipynb     # Main notebook (data prep, training, evaluation)
 ┣ 📜 app.py                    # Streamlit demo app (optional)
 ┣ 📜 requirements.txt          # Dependencies
 ┣ 📜 README.md                 # Project documentation

```

---

## 🔧 How to Run

### 1. Clone the repo
```bash
git clone https://github.com/your-username/smart-resume-screener.git
cd smart-resume-screener
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Jupyter Notebook
```bash
jupyter notebook resume_screener.ipynb
```

### 4. (Optional) Run Streamlit app
```bash
streamlit run app.py
```

---

## 📊 Results

### ✅ Example Evaluation Metrics
- Accuracy: **0.83**  
- Precision: **0.80**  
- Recall: **0.80**  
- F1-Score: **0.80**  

### ✅ Example Predictions
| JD | Resume | Predicted | Probability |
|----|---------|-----------|-------------|
| *Python, ML, AWS* | *ML engineer skilled in Python & AWS* | **Match** | 0.92 |
| *Android Kotlin* | *iOS Swift dev* | **No Match** | 0.12 |


![Alt text](https://github.com/tanishqkolhatkar93/Mentor_Baba_Gen_AI_Resume_Scanner/blob/main/Screenshot%202025-08-28%20131226.png)

---

## 🌱 Next Steps
- 🔹 Try advanced embeddings (Word2Vec, Sentence-BERT, BERT)  
- 🔹 Expand dataset (100+ JD/Resume pairs)  
- 🔹 Hyperparameter tuning & cross-validation  
- 🔹 Integrate into a recruiter dashboard with Streamlit/Flask  

---

## 🙌 MentorBaba Internship Note
This project was built as part of **MentorBaba’s internship assignment**.  
It demonstrates skills in:
- Natural Language Processing (NLP)  
- Machine Learning Classification  
- Practical HR-Tech applications  

---

## 📌 Author
👤 **Tanishq Kolhatkar**  
🔗 [LinkedIn](https://www.linkedin.com/in/tanishq93/) • [GitHub](https://github.com/tanishqkolhatkar93)  
