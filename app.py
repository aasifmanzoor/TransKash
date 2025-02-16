import os
os.system("pip install git+https://github.com/AI4Bharat/IndicTrans.git")

import streamlit as st
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit import IndicProcessor

# Load the model and tokenizer
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

def translate_text(text, src_lang, tgt_lang):
    inputs = tokenizer([text], truncation=True, padding="longest", return_tensors="pt").to(device)
    with torch.no_grad():
        output_tokens = model.generate(**inputs, max_length=256, num_beams=5)
    translation = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]
    return translation

# Streamlit UI
st.title("Kashmiri Translator by Aasif Codes")
st.write("Translate between English and Kashmiri.")

source_lang = st.selectbox("Select Source Language:", ("English", "Kashmiri"))
target_lang = st.selectbox("Select Target Language:", ("English", "Kashmiri"))

text_input = st.text_area("Enter text:")

if st.button("Translate"):
    if text_input.strip():
        src_lang = "eng_Latn" if source_lang == "English" else "kas_Arab"
        tgt_lang = "kas_Arab" if target_lang == "Kashmiri" else "eng_Latn"
        translated_text = translate_text(text_input, src_lang, tgt_lang)
        st.text_area("Translation:", translated_text, height=100)
    else:
        st.warning("Please enter text to translate.")


