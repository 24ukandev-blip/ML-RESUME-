from flask import Flask, request, jsonify
from flask_cors import CORS
import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import base64
import io
import os
import hashlib
import joblib
import tempfile
import PyPDF2
import docx

app = Flask(__name__)
CORS(app)

# Use /tmp for caching (Vercel allows writing to /tmp)
CACHE_FOLDER = '/tmp/cache'
os.makedirs(CACHE_FOLDER, exist_ok=True)

current_model = None
current_vectorizer = None
current_classes = None

def generate_plot_base64():
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    return img_base64

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

def get_file_hash(filepath):
    with open(filepath, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def extract_text_from_bytes(file_bytes, filename):
    ext = os.path.splitext(filename)[1].lower()
    if ext == '.txt':
        return file_bytes.decode('utf-8', errors='ignore')
    elif ext == '.pdf':
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    elif ext == '.docx':
        doc = docx.Document(io.BytesIO(file_bytes))
        return "\n".join([para.text for para in doc.paragraphs])
    else:
        raise ValueError("Unsupported file type")

# ==================== RESUME SKILL ANALYZER ====================
skills_db = {
    "python": ["python", "py"], "java": ["java"], "c++": ["c++", "cpp"], "c": ["c"],
    "javascript": ["javascript", "js"], "typescript": ["typescript", "ts"],
    "go": ["go", "golang"], "rust": ["rust"], "php": ["php"], "ruby": ["ruby"],
    "html": ["html"], "css": ["css"], "react": ["react"], "angular": ["angular"],
    "vue": ["vue"], "node": ["node", "nodejs"], "express": ["express"],
    "machine learning": ["machine learning", "ml"], "deep learning": ["deep learning", "dl"],
    "nlp": ["nlp"], "data analysis": ["data analysis", "analysis"], "pandas": ["pandas"],
    "numpy": ["numpy"], "tensorflow": ["tensorflow"], "pytorch": ["pytorch"],
    "sql": ["sql"], "mysql": ["mysql"], "postgresql": ["postgres", "postgresql"],
    "mongodb": ["mongodb"], "redis": ["redis"], "aws": ["aws"], "azure": ["azure"],
    "gcp": ["gcp"], "docker": ["docker"], "kubernetes": ["kubernetes"], "ci/cd": ["ci/cd"],
    "linux": ["linux"], "django": ["django"], "flask": ["flask"], "spring": ["spring"],
    "fastapi": ["fastapi"], "git": ["git"], "github": ["github"], "jira": ["jira"],
    "android": ["android"], "flutter": ["flutter"], "react native": ["react native"],
    "oop": ["oop"], "data structures": ["dsa", "data structures"], "algorithms": ["algorithms"]
}

job_roles = {
    "Data Scientist": {"python":3,"machine learning":3,"deep learning":2,"pandas":2,"numpy":2,"sql":2},
    "Web Developer": {"html":3,"css":3,"javascript":3,"react":2,"angular":2,"vue":2},
    "Backend Developer": {"python":3,"java":3,"node":3,"django":2,"flask":2,"spring":2,"sql":2,"mongodb":2},
    "Full Stack Developer": {"html":2,"css":2,"javascript":3,"react":2,"node":2,"mongodb":2,"sql":2},
    "C++ Developer": {"c++":3,"c":2,"data structures":3,"algorithms":3},
    "Java Developer": {"java":3,"spring":2,"sql":2},
    "DevOps Engineer": {"docker":3,"kubernetes":3,"aws":2,"ci/cd":2,"linux":2},
    "Cloud Engineer": {"aws":3,"azure":3,"gcp":3,"docker":2},
    "Mobile Developer": {"android":3,"flutter":3,"react native":2}
}

def extract_skills(text):
    text = text.lower()
    words = set(re.findall(r'\b[a-zA-Z\+\#]+\b', text))
    found = set()
    for skill, keywords in skills_db.items():
        for k in keywords:
            if k in words:
                found.add(skill)
    return list(found)

def calculate_scores(found_skills):
    found_set = set(found_skills)
    results = []
    for job, req_skills in job_roles.items():
        total = sum(req_skills.values())
        score = sum(weight for skill, weight in req_skills.items() if skill in found_set)
        percent = int((score / total) * 100) if total else 0
        results.append({"job": job, "score": percent})
    return sorted(results, key=lambda x: x["score"], reverse=True)

@app.route('/api/analyze', methods=['POST'])
def analyze_resume():
    data = request.json
    resume = data.get("resume", "")
    if not resume.strip():
        return jsonify({"error": "Empty resume"}), 400
    found_skills = extract_skills(resume)
    jobs = calculate_scores(found_skills)
    best_job = jobs[0]["job"] if jobs else "Unknown"
    resume_score = jobs[0]["score"] if jobs else 0
    required = job_roles.get(best_job, {})
    missing = [s for s in required if s not in found_skills]
    return jsonify({
        "skills": found_skills,
        "jobs": jobs,
        "best_job": best_job,
        "resume_score": resume_score,
        "missing": missing
    })

@app.route('/api/upload_resume_file', methods=['POST'])
def upload_resume_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if not file.filename.lower().endswith(('.txt', '.pdf', '.docx')):
        return jsonify({"error": "Only .txt, .pdf, .docx files allowed"}), 400

    try:
        file_bytes = file.read()
        text = extract_text_from_bytes(file_bytes, file.filename)
        text = re.sub(r'\s+', ' ', text).strip()
        found_skills = extract_skills(text)
        jobs = calculate_scores(found_skills)
        best_job = jobs[0]["job"] if jobs else "Unknown"
        resume_score = jobs[0]["score"] if jobs else 0
        required = job_roles.get(best_job, {})
        missing = [s for s in required if s not in found_skills]
        return jsonify({
            "skills": found_skills,
            "jobs": jobs,
            "best_job": best_job,
            "resume_score": resume_score,
            "missing": missing
        })
    except Exception as e:
        return jsonify({"error": f"Error processing file: {str(e)}"}), 500

# ==================== DATASET TRAINING & PREDICTION ====================
@app.route('/api/upload_dataset', methods=['POST'])
def upload_dataset():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if not file.filename.endswith('.csv'):
        return jsonify({"error": "Only CSV files allowed"}), 400

    fd, temp_path = tempfile.mkstemp(suffix='.csv')
    os.close(fd)
    file.save(temp_path)
    return jsonify({"message": "File uploaded successfully", "filepath": temp_path})

@app.route('/api/analyze_resume_dataset', methods=['POST'])
def analyze_resume_dataset():
    global current_model, current_vectorizer, current_classes
    data = request.json
    filepath = data.get("filepath")
    if not filepath or not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 400

    file_hash = get_file_hash(filepath)
    cache_path = os.path.join(CACHE_FOLDER, f"{file_hash}.pkl")
    model_cache_path = os.path.join(CACHE_FOLDER, f"{file_hash}_model.pkl")

    if os.path.exists(cache_path) and os.path.exists(model_cache_path):
        print("Loading cached results...")
        cached = joblib.load(cache_path)
        model_data = joblib.load(model_cache_path)
        current_model = model_data['model']
        current_vectorizer = model_data['vectorizer']
        current_classes = model_data['classes']
        return jsonify(cached)

    df = pd.read_csv(filepath, engine='python', on_bad_lines='skip')

    # Auto-detect resume and category columns
    resume_col = None
    cat_col = None
    for col in df.columns:
        if 'resume' in col.lower() and 'html' not in col.lower():
            resume_col = col
        if 'category' in col.lower():
            cat_col = col
    if resume_col is None or cat_col is None:
        return jsonify({"error": "Could not auto-detect 'Resume' and 'Category' columns."}), 400

    df = df[[resume_col, cat_col]].copy()
    df.rename(columns={resume_col: 'Resume', cat_col: 'Category'}, inplace=True)
    df.dropna(inplace=True)
    df['Resume'] = df['Resume'].astype(str).apply(lambda x: re.sub(r'\s+', ' ', x).strip())

    # EDA
    shape = df.shape
    duplicates = int(df.duplicated().sum())
    missing = df.isnull().sum().to_dict()
    categories = df['Category'].value_counts().to_dict()

    # Class distribution plot
    fig, ax = plt.subplots(figsize=(8,5))
    df['Category'].value_counts().plot(kind='bar', ax=ax, color='skyblue')
    ax.set_title('Category Distribution')
    ax.set_xlabel('Category')
    ax.set_ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    class_dist_plot = generate_plot_base64()

    # Resume length distribution
    df['resume_length'] = df['Resume'].apply(lambda x: len(str(x)))
    fig, ax = plt.subplots(figsize=(8,5))
    for cat in df['Category'].unique():
        subset = df[df['Category'] == cat]
        ax.hist(subset['resume_length'], bins=20, alpha=0.5, label=cat)
    ax.set_title('Resume Length Distribution by Category')
    ax.set_xlabel('Length (characters)')
    ax.set_ylabel('Frequency')
    ax.legend()
    length_dist_plot = generate_plot_base64()

    # Preprocessing
    df['clean_resume'] = df['Resume'].apply(clean_text)
    tfidf = TfidfVectorizer(max_features=1500, stop_words='english')
    X = tfidf.fit_transform(df['clean_resume'])
    y = df['Category']
    labels = sorted(y.unique())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    acc_lr = accuracy_score(y_test, y_pred_lr)
    report_lr = classification_report(y_test, y_pred_lr, output_dict=True)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    report_rf = classification_report(y_test, y_pred_rf, output_dict=True)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_lr)
    fig, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=labels, yticklabels=labels)
    ax.set_title('Confusion Matrix (Logistic Regression)')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.xticks(rotation=45, ha='right')
    cm_plot = generate_plot_base64()

    # ROC curves
    roc_plot = None
    if len(labels) <= 10:
        y_test_bin = label_binarize(y_test, classes=labels)
        y_score = lr.predict_proba(X_test)
        fig, ax = plt.subplots(figsize=(8,6))
        for i, label in enumerate(labels):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.2f})')
        ax.plot([0,1], [0,1], 'k--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves (One-vs-Rest)')
        ax.legend(loc='lower right')
        roc_plot = generate_plot_base64()

    # Sample predictions
    sample_texts = df['Resume'].head(5).tolist()
    sample_preds = lr.predict(tfidf.transform(df['clean_resume'].head(5))).tolist()
    samples = [{"text": t[:300] + "...", "predicted": p} for t, p in zip(sample_texts, sample_preds)]

    current_model = lr
    current_vectorizer = tfidf
    current_classes = labels

    results = {
        "shape": shape,
        "duplicates": duplicates,
        "missing": missing,
        "class_distribution": categories,
        "class_dist_plot": class_dist_plot,
        "length_dist_plot": length_dist_plot,
        "accuracy_lr": acc_lr,
        "accuracy_rf": acc_rf,
        "classification_report_lr": report_lr,
        "classification_report_rf": report_rf,
        "confusion_matrix_plot": cm_plot,
        "roc_plot": roc_plot,
        "sample_predictions": samples
    }
    joblib.dump(results, cache_path)
    joblib.dump({'model': lr, 'vectorizer': tfidf, 'classes': labels}, model_cache_path)

    os.unlink(filepath)
    return jsonify(results)

@app.route('/api/predict_single', methods=['POST'])
def predict_single():
    global current_model, current_vectorizer, current_classes
    data = request.json
    resume_text = data.get("resume", "")
    if not resume_text.strip():
        return jsonify({"error": "Empty resume text"}), 400

    if current_model is None:
        cache_files = [f for f in os.listdir(CACHE_FOLDER) if f.endswith('_model.pkl')]
        if cache_files:
            latest = max(cache_files, key=lambda f: os.path.getmtime(os.path.join(CACHE_FOLDER, f)))
            model_data = joblib.load(os.path.join(CACHE_FOLDER, latest))
            current_model = model_data['model']
            current_vectorizer = model_data['vectorizer']
            current_classes = model_data['classes']
        else:
            return jsonify({"error": "No trained model available. Please upload and train a dataset first."}), 400

    cleaned = clean_text(resume_text)
    vec = current_vectorizer.transform([cleaned])
    pred = current_model.predict(vec)[0]
    probs = current_model.predict_proba(vec)[0]
    prob_dict = {current_classes[i]: float(probs[i]) for i in range(len(current_classes))}
    return jsonify({
        "predicted_category": pred,
        "probabilities": prob_dict,
        "confidence": float(max(probs))
    })

# For local testing (optional)
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
