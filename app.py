# app.py
import streamlit as st
from PIL import Image
from config import DEVICE
from image_processor import caption_image
from db_manager import setup_collection, store_image_caption, query_chroma
from query_handler import explain_and_answer

def main():
    st.title("Detailed Image Captioning and Query System")
    st.write(f"Using device: {DEVICE}")
    st.write("Upload an image for a detailed description and ask questions!")

    # Setup ChromaDB collection
    collection = setup_collection()

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            with st.spinner("Generating detailed caption..."):
                caption = caption_image(image)
                st.write("**Detailed Caption:**")
                st.write(caption)
                
                doc_id = f"image_{uploaded_file.name}"
                store_image_caption(collection, doc_id, caption)
                st.success("Caption stored in database!")

            user_query = st.text_input("Ask a question about the image:", "")
            
            if user_query:
                with st.spinner("Processing your query..."):
                    results = query_chroma(collection, user_query, top_k=1)
                    retrieved_caption = results["metadatas"][0][0]["caption"]
                    
                    explanation = explain_and_answer(retrieved_caption, user_query)
                    
                    st.write("**Retrieved Caption:**")
                    st.write(retrieved_caption)
                    st.write("**Answer to Your Query:**")
                    st.write(explanation)
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()