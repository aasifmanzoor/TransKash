import os
os.system("pip install git+https://github.com/AI4Bharat/IndicTrans.git")

import os
import streamlit as st
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit import IndicProcessor

# Install dependencies (Only needed for local deployment)
os.system("pip install git+https://github.com/AI4Bharat/IndicTrans.git")

# Load the model and tokenizer
@st.cache_resource
def load_model():
    ckpt_dir = "ai4bharat/indictrans2-en-indic-1B"
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(ckpt_dir, trust_remote_code=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return tokenizer, model, device

tokenizer, model, device = load_model()
ip = IndicProcessor(inference=True)

# Function for translation
def translate_text(text, src_lang, tgt_lang):
    inputs = tokenizer([text.strip()], truncation=True, padding="longest", return_tensors="pt").to(device)
    
    with torch.no_grad():
        output_tokens = model.generate(**inputs, max_length=512, num_beams=5)
    
    translation = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]
    return translation

# Streamlit UI
st.title("Kashmiri Translator by AasifCodes")
st.write("Translate between **English** and **Kashmiri** (Arabic or Devanagari script).")

# Select Source and Target Languages
source_lang = st.selectbox("Select Source Language:", ["English", "Kashmiri"])
target_lang = st.selectbox("Select Target Language:", ["English", "Kashmiri"])

# Input Text Box
text_input = st.text_area("Enter text:")

# Kashmiri script selection (only if translating to Kashmiri)
script_option = None
if target_lang == "Kashmiri":
    script_option = st.radio("Select Kashmiri Script:", ["Arabic", "Devanagari"])

# Translate Button
if st.button("Translate"):
    if text_input.strip():
        # Assign language codes
        src_lang = "eng_Latn" if source_lang == "English" else "kas_Arab"
        tgt_lang = "kas_Arab" if target_lang == "Kashmiri" else "eng_Latn"

        # If translating to Kashmiri, adjust script
        if target_lang == "Kashmiri":
            tgt_lang = "kas_Arab" if script_option == "Arabic" else "kas_Deva"

        # Debugging - Show selected language codes
        st.write(f"🛠 **Debug:** Source = {src_lang}, Target = {tgt_lang}, Device = {device}")

        # Perform Translation
        translated_text = translate_text(text_input, src_lang, tgt_lang)

        # Display the translated text
        if target_lang == "Kashmiri" and script_option == "Arabic":
            # Apply Noto Nastaliq Urdu font for better readability
            st.markdown(
                """
                <style>
                @import url('https://fonts.googleapis.com/css2?family=Noto+Nastaliq+Urdu&display=swap');
                .translated-text {
                    font-family: 'Noto Nastaliq Urdu', serif;
                    font-size: 20px;
                    direction: rtl; /* Ensures proper alignment */
                }
                </style>
                """,
                unsafe_allow_html=True
            )
            st.markdown(f'<p class="translated-text">{translated_text}</p>', unsafe_allow_html=True)
        else:
            st.text_area("Translation:", translated_text, height=100)
    else:
        st.warning("⚠️ Please enter text to translate.")
