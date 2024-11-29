import streamlit as st
import pytesseract
from PIL import Image
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
import os

# Ensure the required NLTK resources are downloaded
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)  # Use the correct resource

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Streamlit app code
st.title("Image to Text Generator")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the image and display it
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Extract text from the image
    try:
        raw_text = pytesseract.image_to_string(image)
        st.write("Extracted Text:")
        st.write(raw_text)


    except Exception as e:
        st.error(f"Error extracting text from image: {e}")

    # Tokenize and filter words
    if raw_text:
        words = word_tokenize(raw_text)
        filtered_words = [word for word in words if word.lower() not in stopwords.words('english')]

        # Tagging parts of speech
        pos_tags = pos_tag(filtered_words)  # Correct usage
        st.write("POS Tags:")
        st.write(pos_tags)

        # Display a simple summary
        from nltk.tokenize import sent_tokenize

        sentences = sent_tokenize(raw_text)
        summary = ' '.join(sentences[:2])  # Extract the first few sentences
        st.write("### Summary:")
        st.write(summary)
    else:
        st.warning("No text extracted from the image.")
