
## Image Captioning and Query System

A Streamlit app that generates detailed captions for uploaded images using the Idefics2 model, stores them in ChromaDB, and answers user queries based on the captions using a language model.

## Setup
1. Clone the repository:
   git clone <repository-url>
   cd <repository-name>
2. Install dependencies:
   pip install -r requirements.txt
3. Run the app:
   streamlit run app.py

## Features
- Upload images (JPG, JPEG, PNG)
- Generate detailed captions using Idefics2
- Store captions in ChromaDB
- Query captions and get detailed explanations

## Dependencies
See `requirements.txt` for a full list.

## Notes
- Requires a GPU with CUDA support for optimal performance (falls back to CPU if unavailable).
- Uses the `llama3.1` model from Ollama for query explanations.
