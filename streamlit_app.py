
import streamlit as st
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

st.set_page_config(
    page_title="Conversation Summarizer",
    page_icon="📝"
)

st.title("📝 Conversation Summarizer")
st.write("Paste a conversation below to generate a summary.")

MODEL_ID = "Divy1812/conversation_summarizer_t5"


@st.cache_resource
def load_model():
    tokenizer = T5Tokenizer.from_pretrained(MODEL_ID)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_ID)
    model.eval()
    return tokenizer, model


# Load the trained model and tokenizer
with st.spinner("Loading the summarization model..."):
    tokenizer, model = load_model()

st.success("Model loaded successfully!")

# Input area for the conversation
dialogue = st.text_area(
    "Enter your conversation",
    height=250,
    placeholder="Paste your conversation here..."
)

if st.button("Summarize"):
    if not dialogue.strip():
        st.warning("Please enter a conversation first.")
    else:
        with st.spinner("Generating summary..."):
            inputs = tokenizer(
                dialogue,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=150
                )

            summary = tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )

        st.subheader("Generated Summary")
        st.write(summary)
