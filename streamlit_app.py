
import streamlit as st
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Page configuration
st.set_page_config(
    page_title="Conversation Summarizer",
    page_icon="📝",
    layout="wide"
)

# Minimal custom CSS for dark theme
st.markdown("""
<style>
.stApp {
    background-color: #0E1117;
    color: #FAFAFA;
}
.stTextArea textarea {
    background-color: #1C2333;
    color: #FAFAFA;
}
.stButton button {
    background-color: #4F46E5;
    color: white;
    border: none;
}
</style>
""", unsafe_allow_html=True)

# App heading
st.title("📝 Conversation Summarizer")
st.write(
    "Turn lengthy conversations into clear, concise summaries "
    "using a fine-tuned T5 model."
)

st.divider()

# Load model once and cache it
MODEL_ID = "Divy1812/conversation_summarizer_t5"

@st.cache_resource
def load_model():
    tokenizer = T5Tokenizer.from_pretrained(MODEL_ID)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_ID)
    model.eval()
    return tokenizer, model

# Conversation input
st.subheader("Enter your conversation")

dialogue = st.text_area(
    "Conversation",
    placeholder="Paste your conversation here...",
    height=250,
    label_visibility="collapsed"
)

st.caption(f"Words entered: {len(dialogue.split())}")

# Generate summary
if st.button("✨ Summarize Conversation", type="primary"):
    if not dialogue.strip():
        st.warning("Please enter a conversation first.")
    else:
        try:
            with st.spinner("Generating summary..."):
                tokenizer, model = load_model()

                inputs = tokenizer(
                    dialogue,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                )

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=180,
                        num_beams=4,
                        early_stopping=True,
                        no_repeat_ngram_size=3
                    )

                summary = tokenizer.decode(
                    outputs[0],
                    skip_special_tokens=True
                ).strip()

            st.subheader("Generated Summary")

            if summary:
                st.write(summary)

                st.caption(
                    f"Words in summary: {len(summary.split())}"
                )

                st.download_button(
                    "Download Summary",
                    data=summary,
                    file_name="conversation_summary.txt",
                    mime="text/plain"
                )
            else:
                st.warning("The model returned an empty summary.")

        except Exception:
            st.error(
                "Could not generate the summary. "
                "Please check the application logs."
            )

st.divider()
st.caption("Built with Streamlit, PyTorch, and Hugging Face Transformers.")

