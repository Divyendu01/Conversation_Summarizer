
# Conversation Summarizer using Fine-Tuned T5

A web application that generates concise summaries of lengthy conversations using a fine-tuned T5-small transformer model.

## Live Demo
[Try the Conversation Summarizer](https://divyendu-conversation-summarizer.streamlit.app)

## Features
- Generate concise summaries from text conversations
- Fine-tuned T5 model for conversation summarization
- Word counts for input and generated summary
- Download summaries as text files
- Web interface built with Streamlit

## Technologies Used
- Python
- PyTorch
- Hugging Face Transformers
- T5-small
- Streamlit

## Model
The fine-tuned model is hosted on Hugging Face:
[Divy1812/conversation_summarizer_t5](https://huggingface.co/Divy1812/conversation_summarizer_t5)

## Run Locally

1. Clone the repository:
   `git clone https://github.com/Divyendu01/Conversation_Summarizer.git`

2. Install dependencies:
   `pip install -r requirements.txt`

3. Launch the app:
   `streamlit run streamlit_app.py`
