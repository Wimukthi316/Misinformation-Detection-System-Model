import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK data if not present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Initialize NLTK tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Load the model and vectorizer
@st.cache_resource
def load_model():
    model = joblib.load('models/best_misinfo_detection_model.joblib')
    vectorizer = joblib.load('models/tfidf_vectorizer.joblib')
    metadata = joblib.load('models/model_metadata.joblib')
    return model, vectorizer, metadata

def advanced_text_cleaning(text):
    """
    Advanced text cleaning function for misinformation detection.
    Same as used in training.
    """
    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)

    # Remove mentions
    text = re.sub(r'@\w+', '', text)

    # Remove hashtags but keep the word
    text = re.sub(r'#', '', text)

    # Remove special characters and numbers, keep only alphabets and spaces
    text = re.sub(r'[^a-z\s]', '', text)

    # Tokenize
    tokens = text.split()

    # Remove stopwords and short words
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]

    # Lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return " ".join(tokens)

def predict_misinformation(text, model, vectorizer, threshold=0.6):
    """
    Predict whether a given text is misinformation or not.
    """
    # Clean the input text
    cleaned = advanced_text_cleaning(text)

    if not cleaned:
        return {
            'status': 'error',
            'message': 'No meaningful content found after cleaning.'
        }

    # Vectorize
    text_tfidf = vectorizer.transform([cleaned])

    # Predict
    prediction = model.predict(text_tfidf)[0]

    # Get probability if available
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(text_tfidf)[0]
        confidence = probabilities[prediction]
    else:
        confidence = 1.0

    # Determine verdict
    if prediction == 1:
        if confidence >= threshold:
            verdict = "🚨 FAKE NEWS"
            color = "red"
        else:
            verdict = "⚠️  LIKELY FAKE"
            color = "orange"
    else:
        if confidence >= threshold:
            verdict = "✅ REAL NEWS"
            color = "green"
        else:
            verdict = "🤔 LIKELY REAL"
            color = "lightgreen"

    return {
        'status': 'success',
        'prediction': int(prediction),
        'verdict': verdict,
        'confidence': confidence,
        'color': color,
        'cleaned_text': cleaned
    }

# Streamlit App
def main():
    st.set_page_config(
        page_title="Misinformation Detection System",
        page_icon="📰",
        layout="wide"
    )

    # Title and description
    st.title("📰 Misinformation Detection System")
    st.markdown("""
    **Detect fake news and misinformation in real-time!**

    This AI-powered system analyzes news articles and social media posts to determine their authenticity.
    Enter any news text below and get an instant analysis.
    """)

    # Load model
    with st.spinner("Loading AI model..."):
        model, vectorizer, metadata = load_model()

    # Display model info
    st.sidebar.header("🤖 Model Information")
    st.sidebar.write(f"**Model:** {metadata['model_name']}")
    st.sidebar.write(f"**Accuracy:** {metadata['accuracy']:.3f}")
    st.sidebar.write(f"**F1-Score:** {metadata['f1_score']:.3f}")
    st.sidebar.write(f"**Training Samples:** {metadata['training_samples']:,}")
    st.sidebar.write(f"**Vocabulary Size:** {metadata['vocabulary_size']:,}")

    # Main input area
    st.header("📝 Enter News Text for Analysis")

    # Text input
    user_input = st.text_area(
        "Paste your news article or social media post here:",
        height=200,
        placeholder="Enter the news text you want to analyze for misinformation..."
    )

    # Prediction button
    if st.button("🔍 Analyze Text", type="primary", use_container_width=True):
        if user_input.strip():
            with st.spinner("Analyzing text..."):
                result = predict_misinformation(user_input, model, vectorizer)

            if result['status'] == 'error':
                st.error(f"❌ {result['message']}")
            else:
                # Display results
                st.success("Analysis Complete!")

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("📊 Prediction Results")

                    # Verdict with color
                    if result['color'] == 'red':
                        st.error(result['verdict'])
                    elif result['color'] == 'orange':
                        st.warning(result['verdict'])
                    elif result['color'] == 'green':
                        st.success(result['verdict'])
                    else:
                        st.info(result['verdict'])

                    st.write(f"**Confidence:** {result['confidence']:.1%}")
                    st.write(f"**Classification:** {'Fake News (1)' if result['prediction'] == 1 else 'Real News (0)'}")

                with col2:
                    st.subheader("🔧 Processed Text")
                    st.write("**Cleaned Text:**")
                    st.text_area("", result['cleaned_text'], height=100, disabled=True)

                # Additional info
                st.subheader("ℹ️ How it Works")
                st.markdown("""
                1. **Text Cleaning**: Removes URLs, mentions, special characters, and stopwords
                2. **Feature Extraction**: Converts text to numerical features using TF-IDF
                3. **AI Prediction**: Uses ensemble machine learning model for classification
                4. **Confidence Scoring**: Provides probability-based confidence levels
                """)

        else:
            st.warning("⚠️ Please enter some text to analyze.")

    # Footer
    st.markdown("---")
    st.markdown("""
    **Built with ❤️ using Streamlit and Machine Learning**

    *Disclaimer: This tool provides AI-assisted analysis but should not be the sole basis for fact-checking.
    Always verify information from multiple reliable sources.*
    """)

if __name__ == "__main__":
    main()