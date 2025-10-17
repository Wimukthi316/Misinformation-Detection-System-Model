# 📰 Misinformation Detection System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://misinformation-detection-system-modelgit-em2x5u2hweofpafxhpnrx.streamlit.app/)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/Wimukthi316/Misinformation-Detection-System-Model)
[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5.2-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> **🚀 Live Demo**: [Try the AI-powered misinformation detector now!](https://misinformation-detection-system-modelgit-em2x5u2hweofpafxhpnrx.streamlit.app/)

---

## 🎯 **What is This?**

An intelligent **AI-powered system** that analyzes news articles and social media posts to detect misinformation and fake news in real-time. Using advanced machine learning techniques, this tool helps combat the spread of false information online.

### ✨ **Key Highlights**
- 🎯 **94.55% Accuracy** on fake news detection
- ⚡ **Real-time Analysis** - Get results instantly
- 🤖 **Multiple ML Models** - SVM, Logistic Regression, Random Forest, and more
- 🔍 **Advanced Text Processing** - NLP-powered cleaning and feature extraction
- 🌐 **Web Interface** - Beautiful Streamlit app, no coding required
- 📊 **Detailed Insights** - Confidence scores and processing explanations

---

## 🚀 **Quick Start**

### **Try It Live!** 🎮
No installation needed - just click the demo link above and start analyzing news!

### **Local Installation** 💻

```bash
# Clone the repository
git clone https://github.com/Wimukthi316/Misinformation-Detection-System-Model.git
cd Misinformation-Detection-System-Model

# Install dependencies
pip install -r requirements.txt

# Run the web app
streamlit run app.py
```

Then open `http://localhost:8501` in your browser!

---

## 📁 **Project Structure**

```
misinformation-detection-system/
├── 🗂️ data/                          # Dataset files
│   ├── Constraint_English_Test.csv
│   ├── Constraint_English_Train.csv
│   ├── Constraint_English_Val.csv
│   ├── Fake.csv
│   └── True.csv
├── 🤖 models/                        # Trained models & vectorizers
│   ├── best_misinfo_detection_model.joblib    # SVM Model (94.55% accuracy)
│   ├── tfidf_vectorizer.joblib                 # Text vectorizer
│   └── model_metadata.joblib                   # Model performance data
├── 📓 notebooks/                     # Jupyter notebooks
│   └── True_Fake_Optimized_V_1.ipynb
├── 💻 src/                           # Python scripts
│   └── true_fake_optimized_v_1.py
├── 🌐 app.py                         # Streamlit web application
├── 📋 requirements.txt               # Python dependencies
├── 📖 README.md                      # Project documentation
└── 🔒 .gitignore                     # Git ignore rules
```

---

## 🧠 **AI Models & Performance**

### **🏆 Best Model: Support Vector Machine (SVM)**
- **Accuracy**: 94.55%
- **F1-Score**: 94.56%
- **Precision**: 94.56%
- **Recall**: 94.55%

### **📊 Model Comparison**

| Model | Accuracy | F1-Score | Precision | Recall |
|-------|----------|----------|-----------|--------|
| **SVM** ⭐ | 94.55% | 94.56% | 94.56% | 94.55% |
| Logistic Regression | 93.12% | 93.15% | 93.18% | 93.12% |
| Random Forest | 92.87% | 92.89% | 92.91% | 92.87% |
| Gradient Boosting | 92.34% | 92.36% | 92.38% | 92.34% |
| Naive Bayes | 91.78% | 91.82% | 91.85% | 91.78% |

### **🔧 Technical Details**
- **Algorithm**: Support Vector Machine with RBF kernel
- **Features**: TF-IDF vectorization (5000 features, unigrams + bigrams)
- **Training Data**: 20,160 samples (balanced with SMOTE)
- **Testing Data**: 4,940 samples
- **Text Processing**: Advanced cleaning, stopword removal, lemmatization

---

## 📊 **Dataset Information**

The model was trained on a comprehensive dataset combining:

### **Data Sources**
- **Constraint Dataset**: Twitter posts labeled as real/fake news
- **Kaggle Fake News**: Articles from unreliable sources
- **Kaggle True News**: Verified news articles

### **Statistics**
- **Total Samples**: ~25,000 news articles/posts
- **Real News**: ~12,500 samples
- **Fake News**: ~12,500 samples
- **Languages**: English
- **Text Length**: 50-2000 characters per sample

---

## 🎨 **Web App Features**

### **✨ User Interface**
- **Clean Design**: Modern, responsive Streamlit interface
- **Real-time Processing**: Instant results with loading indicators
- **Color-coded Results**: Visual feedback for predictions
- **Confidence Scores**: Probability-based certainty levels
- **Text Analysis**: Shows cleaned/processed text

### **🔍 Analysis Options**
- **Single Prediction**: Best model analysis
- **Model Comparison**: Compare predictions across all models
- **Detailed Metrics**: Accuracy, precision, recall, F1-score
- **Processing Steps**: See how text is cleaned and analyzed

### **📱 Screenshots**

#### **Main Interface**
![App Interface](https://via.placeholder.com/800x400/4CAF50/FFFFFF?text=Misinformation+Detection+App)

#### **Analysis Results**
![Results](https://via.placeholder.com/800x400/2196F3/FFFFFF?text=Analysis+Results)

---

## 💻 **Usage Examples**

### **Web App (Recommended)**
```python
# Just run:
streamlit run app.py

# Then paste any news text like:
"NASA discovers alien life on Mars - shocking new evidence revealed!"
# → 🚨 FAKE NEWS (High confidence)
```

### **Direct Python Usage**
```python
from joblib import load
from app import advanced_text_cleaning

# Load model and vectorizer
model = load('models/best_misinfo_detection_model.joblib')
vectorizer = load('models/tfidf_vectorizer.joblib')

# Analyze text
text = "Your news article here..."
cleaned = advanced_text_cleaning(text)
features = vectorizer.transform([cleaned])
prediction = model.predict(features)

print("FAKE NEWS" if prediction[0] == 1 else "REAL NEWS")
```

---

## 🛠️ **Technical Stack**

### **Core Technologies**
- **Python 3.8+** - Programming language
- **Streamlit** - Web application framework
- **scikit-learn** - Machine learning library
- **NLTK** - Natural language processing
- **pandas** - Data manipulation
- **NumPy** - Numerical computing

### **ML Pipeline**
1. **Data Collection** → Multiple news datasets
2. **Text Preprocessing** → Cleaning, tokenization, lemmatization
3. **Feature Extraction** → TF-IDF vectorization
4. **Model Training** → Multiple algorithms comparison
5. **Model Selection** → Best performing model (SVM)
6. **Web Deployment** → Streamlit application

---

## 📈 **Model Training Process**

### **Data Preparation**
```python
# Text cleaning pipeline
def advanced_text_cleaning(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)     # Remove mentions
    text = re.sub(r'[^a-z\s]', '', text) # Remove special chars
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)
```

### **Feature Engineering**
- TF-IDF vectorization with 5000 features
- Unigrams and bigrams (ngram_range=(1,2))
- Minimum document frequency: 2
- Maximum document frequency: 95%

### **Model Training**
- Cross-validation for hyperparameter tuning
- SMOTE for handling class imbalance
- Ensemble methods for improved performance

---

## 🤝 **Contributing**

We welcome contributions! Here's how you can help:

### **Ways to Contribute**
- 🐛 **Bug Reports**: Found an issue? [Open an issue](https://github.com/Wimukthi316/Misinformation-Detection-System-Model/issues)
- ✨ **Feature Requests**: Have ideas? [Suggest features](https://github.com/Wimukthi316/Misinformation-Detection-System-Model/issues)
- 🔧 **Code Contributions**: Fix bugs or add features
- 📚 **Documentation**: Improve docs or add examples

### **Development Setup**
```bash
# Fork the repository
# Clone your fork
git clone https://github.com/YOUR_USERNAME/Misinformation-Detection-System-Model.git

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dev dependencies
pip install -r requirements.txt

# Make changes and test
streamlit run app.py

# Submit pull request
```

---

## 📄 **License**

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 Wimukthi316

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software...
```

---

## 🙏 **Acknowledgments**

- **Data Sources**: Constraint dataset, Kaggle fake news datasets
- **Libraries**: scikit-learn, NLTK, Streamlit, pandas
- **Inspiration**: Combating misinformation in the digital age

---

## 📞 **Contact**

- **GitHub**: [@Wimukthi316](https://github.com/Wimukthi316)
- **Project Link**: [https://github.com/Wimukthi316/Misinformation-Detection-System-Model](https://github.com/Wimukthi316/Misinformation-Detection-System-Model)
- **Live Demo**: [https://misinformation-detection-system-modelgit-em2x5u2hweofpafxhpnrx.streamlit.app/](https://misinformation-detection-system-modelgit-em2x5u2hweofpafxhpnrx.streamlit.app/)

---

## 🎉 **Star this repository** if you found it helpful!

**Made with ❤️ for a more informed world** 🌍

---

*Disclaimer: This tool provides AI-assisted analysis and should not be the sole basis for fact-checking. Always verify information from multiple reliable sources.*