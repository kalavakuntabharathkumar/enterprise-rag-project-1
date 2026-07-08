import streamlit as st
import requests
import os
from pathlib import Path

# Backend URL - override with BACKEND_URL when the backend isn't on
# localhost (e.g. the "backend" service name in docker-compose).
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# If the backend was started with API_KEY set (see backend/main.py's auth
# stub), the frontend needs to send the same value back as X-API-Key or
# every request gets a 401. Leave API_KEY unset on both sides for local,
# unauthenticated use.
API_KEY = os.getenv("API_KEY")
AUTH_HEADERS = {"X-API-Key": API_KEY} if API_KEY else {}

st.set_page_config(page_title="Enterprise AI Document Assistant", page_icon="📄")

st.title("📄 Enterprise AI Document Assistant")
st.markdown("Upload PDF documents and ask questions to get AI-powered answers with source citations.")

# File upload
st.header("1. Upload PDF Document")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    if st.button("Process Document"):
        with st.spinner("Processing document..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
                response = requests.post(f"{BACKEND_URL}/upload", files=files, headers=AUTH_HEADERS)
                if response.status_code == 200:
                    st.success("Document processed successfully!")
                    st.session_state.document_uploaded = True
                else:
                    st.error(f"Error processing document: {response.text}")
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Question answering
st.header("2. Ask Questions")
question = st.text_input("Enter your question about the document:")

if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Generating answer..."):
            try:
                response = requests.post(f"{BACKEND_URL}/ask", json={"question": question}, headers=AUTH_HEADERS)
                if response.status_code == 200:
                    result = response.json()
                    st.subheader("Answer:")
                    st.write(result["answer"])
                    st.subheader("Confidence Score:")
                    st.write(f"{result['confidence']:.2f}")
                    if result.get("intent"):
                        st.caption(f"Detected intent: {result['intent']}")
                    if result["sources"]:
                        st.subheader("Sources:")
                        for source in result["sources"]:
                            st.write(f"- {Path(source).name}")
                else:
                    st.error(f"Error: {response.text}")
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Instructions
st.header("Instructions")
st.markdown("""
1. Upload a PDF document using the file uploader above.
2. Click "Process Document" to extract text, generate embeddings, and store in the vector database.
3. Enter your question in the text input field.
4. Click "Ask" to get an AI-generated answer with source citations and confidence score.
5. Make sure the backend is running on http://localhost:8000
""")

# Footer
st.markdown("---")
st.markdown("Built with FastAPI, Streamlit, LangChain, and OpenAI.")