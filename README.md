
# Conversation Summarizer using T5 Transformer

A deep learning project that summarizes conversations using a fine-tuned T5 (Text-to-Text Transfer Transformer) model, integrated with a FastAPI web application.

## Approach

- Conversation preprocessing and cleaning using regular expressions
- Tokenization using the T5 tokenizer
- Conversation summarization using a fine-tuned T5 model from Hugging Face Transformers
- Summary generation using beam search
- FastAPI endpoints for conversation summarization
- Interactive web interface using HTML, CSS, and JavaScript

## Dataset

The project uses a fine-tuned T5 model saved locally in the `saved_summary_model` directory. The application accepts user-provided conversations as input and generates concise summaries of the discussion.

## Results

Developed a web-based conversation summarization application that accepts conversation text and generates concise summaries using a fine-tuned T5 Transformer model.

## Tech Stack

Python · PyTorch · Hugging Face Transformers · FastAPI · REST APIs · HTML · CSS · JavaScript · Jinja2 · Uvicorn

## Run

1. Clone the repository and navigate to the project directory.
2. Install the required dependencies:

   ```bash
   pip install fastapi uvicorn transformers torch sentencepiece jinja2
   ```

3. Ensure the `saved_summary_model` directory contains the fine-tuned model and tokenizer files.
4. Start the FastAPI application:

   ```bash
   python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
   ```

5. Open `http://127.0.0.1:8000/` in your browser to use the application.
6. Alternatively, visit `http://127.0.0.1:8000/docs` to test the API endpoints.

## Author

**Divyendu Sharma**

