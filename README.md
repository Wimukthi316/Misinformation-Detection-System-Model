# Misinformation Detection System

## Overview
This project implements a machine learning-based system for detecting misinformation in news articles and social media posts. The system uses natural language processing techniques and various classification algorithms to distinguish between real and fake news.

## Features
- **Text Preprocessing**: Advanced text cleaning, tokenization, and feature engineering
- **Multiple ML Models**: Logistic Regression, Naive Bayes, SVM, Random Forest, Gradient Boosting, and Ensemble methods
- **Data Balancing**: SMOTE for handling class imbalance
- **Visualization**: EDA plots, word clouds, and model performance comparisons
- **Interactive Prediction**: Real-time misinformation detection interface

## Dataset
The model is trained on a combination of datasets:
- Constraint English dataset (tweets)
- Kaggle Fake News dataset
- Kaggle True News dataset

## Model Performance
- **Best Model**: Ensemble (Voting Classifier)
- **F1-Score**: [Insert F1 score from your results]
- **Accuracy**: [Insert accuracy from your results]

## Files Description
- `True_Fake_Optimized_V_1.ipynb`: Main Jupyter notebook with complete implementation
- `true_fake_optimized_v_1.py`: Python script version
- `best_misinfo_detection_model.joblib`: Trained best model
- `tfidf_vectorizer.joblib`: TF-IDF vectorizer for text processing
- `model_metadata.joblib`: Model metadata and performance metrics
- CSV files: Training datasets

## Installation
```bash
pip install -r requirements.txt
```

## Usage
### Training
Run the Jupyter notebook `True_Fake_Optimized_V_1.ipynb` to train the models.

### Prediction
```python
from joblib import load
import pandas as pd

# Load model and vectorizer
model = load('best_misinfo_detection_model.joblib')
vectorizer = load('tfidf_vectorizer.joblib')

# Example prediction
text = "Your news text here"
# Preprocess text (use the cleaning function from the notebook)
# Then vectorize and predict
```

## Requirements
- Python 3.7+
- scikit-learn
- nltk
- imbalanced-learn
- wordcloud
- matplotlib
- seaborn
- pandas
- numpy

## Results
[Include some key visualizations or results from your notebook]

## Contributing
Feel free to contribute by opening issues or submitting pull requests.

## License
[Specify license if any]