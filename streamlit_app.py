
import streamlit as st
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Conversation Summarizer",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #f5f7fb;
}

.block-container {
    max-width: 1120px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Header */
.hero {
    background: linear-gradient(125deg, #172554, #3730a3, #6d28d9);
    padding: 42px 40px;
    border-radius: 24px;
    color: white;
    margin-bottom: 28px;
    box-shadow: 0 12px 35px rgba(55, 48, 163, 0.16);
}

.hero-tag {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.25);
    padding: 6px 12px;
    border-radius: 30px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.5px;
    margin-bottom: 16px;
}

.hero h1 {
    font-size: 38px;
    line-height: 1.2;
    font-weight: 800;
    margin: 0 0 12px 0;
    color: white;
}

.hero p {
    font-size: 16px;
    line-height: 1.7;
    color: #e0e7ff;
    margin: 0;
    max-width: 650px;
}

/* Section headings */
.section-heading {
    font-size: 20px;
    font-weight: 700;
    color: #172554;
    margin: 8px 0 5px 0;
}

.section-description {
    color: #64748b;
    font-size: 14px;
    margin-bottom: 16px;
}

/* Input and output areas */
textarea {
    border-radius: 14px !important;
}

div[data-testid="stTextArea"] textarea {
    border: 1px solid #dbe2ef;
    background: white;
    border-radius: 14px;
    padding: 16px;
    font-size: 14px;
    line-height: 1.7;
}

div[data-testid="stTextArea"] textarea:focus {
    border-color: #6366f1;
    box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.12);
}

/* Buttons */
.stButton > button {
    background: linear-gradient(100deg, #4f46e5, #6d28d9);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 12px 20px;
    min-height: 48px;
    font-weight: 700;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    color: white;
    border: none;
    box-shadow: 0 6px 18px rgba(79, 70, 229, 0.25);
    transform: translateY(-1px);
}

/* Output panel */
.summary-panel {
    background: white;
    border: 1px solid #e0e7ff;
    border-left: 5px solid #6366f1;
    border-radius: 14px;
    padding: 22px 24px;
    color: #1e293b;
    font-size: 15px;
    line-height: 1.9;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    min-height: 110px;
}

.info-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 15px;
    padding: 18px;
    margin-bottom: 12px;
}

.info-card h4 {
    color: #334155;
    font-size: 14px;
    margin: 0 0 8px 0;
}

.info-card p {
    color: #64748b;
    font-size: 13px;
    line-height: 1.6;
    margin: 0;
}

.footer {
    text-align: center;
    color: #94a3b8;
    font-size: 12px;
    padding-top: 30px;
}

/* Mobile layout */
@media (max-width: 640px) {
    .hero {
        padding: 28px 22px;
    }

    .hero h1 {
        font-size: 29px;
    }

    .hero p {
        font-size: 14px;
    }

    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }
}
</style>
""", unsafe_allow_html=True)


# ---------------- HEADER ----------------

st.markdown("""
<div class="hero">
    <div class="hero-tag">✦ AI-POWERED WRITING TOOL</div>
    <h1>Conversation Summarizer</h1>
    <p>
        Turn lengthy conversations into clear, concise summaries.
        Paste your discussion below and let AI highlight the key information.
    </p>
</div>
""", unsafe_allow_html=True)


# ---------------- LOAD MODEL ----------------

MODEL_ID = "Divy1812/conversation_summarizer_t5"


@st.cache_resource
def load_model():
    tokenizer = T5Tokenizer.from_pretrained(MODEL_ID)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_ID)
    model.eval()
    return tokenizer, model


# ---------------- MAIN LAYOUT ----------------

input_col, tips_col = st.columns([2.1, 1], gap="large")

with input_col:
    st.markdown(
        '<div class="section-heading">Your conversation</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        'Paste a meeting transcript, chat, or discussion below.'
        '</div>',
        unsafe_allow_html=True
    )

    dialogue = st.text_area(
        "Conversation input",
        height=300,
        placeholder=(
            "Example:\n"
            "Rahul: We need to finish the project by Friday.\n"
            "Priya: I have completed the first two sections.\n"
            "Rahul: Great. I will review the final version on Thursday."
        ),
        label_visibility="collapsed"
    )

    word_count = len(dialogue.split()) if dialogue.strip() else 0

    st.caption(f"📝 {word_count} words entered")

    summarize_clicked = st.button(
        "✦  Summarize Conversation",
        use_container_width=True
    )

with tips_col:
    st.markdown("""
    <div class="info-card">
        <h4>✦ How it works</h4>
        <p>
            1. Paste your conversation.<br><br>
            2. Click Summarize Conversation.<br><br>
            3. Read and save your generated summary.
        </p>
    </div>

    <div class="info-card">
        <h4>💡 Best results</h4>
        <p>
            Include complete sentences and speaker names.
            For long conversations, focus on the most relevant discussion.
        </p>
    </div>

    <div class="info-card">
        <h4>🤖 Powered by T5</h4>
        <p>
            Uses your fine-tuned T5 model hosted on Hugging Face.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ---------------- SUMMARIZATION ----------------

if summarize_clicked:
    if not dialogue.strip():
        st.warning("Please enter a conversation before summarizing.")

    else:
        try:
            with st.spinner("Loading model and generating your summary..."):
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
                        max_new_tokens=100,
                        num_beams=4,
                        early_stopping=True
                    )

                summary = tokenizer.decode(
                    outputs[0],
                    skip_special_tokens=True
                ).strip()

            st.markdown("---")

            st.markdown(
                '<div class="section-heading">✨ Generated Summary</div>',
                unsafe_allow_html=True
            )

            if summary:
                st.markdown(
                    '<div class="summary-panel">'
                    + summary.replace("&", "&amp;")
                             .replace("<", "&lt;")
                             .replace(">", "&gt;")
                    + '</div>',
                    unsafe_allow_html=True
                )

                summary_words = len(summary.split())
                st.caption(f"📄 {summary_words} words in summary")

                st.download_button(
                    label="⬇️ Download Summary",
                    data=summary,
                    file_name="conversation_summary.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            else:
                st.warning("The model returned an empty summary. Please try again.")

        except Exception:
            st.error(
                "We couldn't generate the summary. "
                "Please check the app logs for the underlying error."
            )


# ---------------- FOOTER ----------------

st.markdown("""
<div class="footer">
    Conversation Summarizer · Built with Streamlit and T5
</div>
""", unsafe_allow_html=True)
