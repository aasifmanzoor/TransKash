import os
os.system("pip install git+https://github.com/AI4Bharat/IndicTrans.git")
import streamlit as st
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit import IndicProcessor
import pyttsx3

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

def translate_text(text, target_script):
    src_lang = "eng_Latn"
    tgt_lang = "kas_Arab" if target_script == "Arabic" else "kas_Deva"
    inputs = tokenizer([text], truncation=True, padding="longest", return_tensors="pt").to(device)
    with torch.no_grad():
        output_tokens = model.generate(**inputs, max_length=256, num_beams=5)
    translation = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]
    return translation

def text_to_speech(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# Streamlit UI
st.title("Kashmiri Translator by AasifCodes")
st.write("Translate English text into Kashmiri (Arabic or Devanagari script) with TTS support.")

text_input = st.text_area("Enter text in English:")
script_option = st.radio("Select Kashmiri Script:", ("Arabic", "Devanagari"))

if st.button("Translate"):
    if text_input.strip():
        translated_text = translate_text(text_input, script_option)
        st.success(f"Translation ({script_option}): {translated_text}")
        if st.button("🔊 Listen to Translation"):
            text_to_speech(translated_text)
    else:
        st.warning("Please enter text to translate.")
