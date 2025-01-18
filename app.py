# app.py

import os
import tempfile
from typing import List, Dict, Any
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS  # We'll still use FAISS but through a different package
from langchain.chains import ConversationalRetrievalChain
import google.generativeai as genai
from pptx import Presentation
import docx2txt
import numpy as np

class PowerPointLoader:
    """Custom PowerPoint loader that extracts text without relying on unstructured"""
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> List[Dict[str, Any]]:
        prs = Presentation(self.file_path)
        text_content = []
        
        for slide in prs.slides:
            slide_text = []
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    slide_text.append(shape.text)
            
            if slide_text:
                text_content.append({
                    "page_content": "\n".join(slide_text),
                    "metadata": {"source": self.file_path}
                })
        
        return text_content

class DocumentManager:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        self.processed_files = []
        self.qa_chain = None
        self.chat_history = []
        
    def process_file(self, uploaded_file) -> List[Dict[str, Any]]:
        """Process a single uploaded file and return its chunks"""
        try:
            # Create a temporary file to store the uploaded content
            with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                file_path = tmp_file.name

            if uploaded_file.name.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
                documents = loader.load()
            elif uploaded_file.name.endswith('.docx'):
                text = docx2txt.process(file_path)
                documents = [{
                    "page_content": text,
                    "metadata": {"source": uploaded_file.name}
                }]
            elif uploaded_file.name.endswith(('.pptx', '.ppt')):
                loader = PowerPointLoader(file_path)
                documents = loader.load()
            else:
                st.error(f"Unsupported file type: {uploaded_file.name}")
                return []

            chunks = self.text_splitter.create_documents(
                texts=[doc["page_content"] for doc in documents],
                metadatas=[doc["metadata"] for doc in documents]
            )
            
            self.processed_files.append(uploaded_file.name)
            os.unlink(file_path)  # Clean up temporary file
            return chunks
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
            return []

    def setup_qa_system(self, files) -> bool:
        """Initialize the QA system with uploaded documents"""
        try:
            all_chunks = []
            for file in files:
                chunks = self.process_file(file)
                all_chunks.extend(chunks)

            if not all_chunks:
                st.error("No documents were successfully processed!")
                return False

            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            
            vectorstore = FAISS.from_documents(
                all_chunks,
                embeddings
            )
            
            self.qa_chain = ConversationalRetrievalChain.from_llm(
                ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash",
                    temperature=0.7,
                    convert_system_message_to_human=True
                ),
                vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 4}  # Adjust number of retrieved documents
                ),
                return_source_documents=True
            )
            return True
        except Exception as e:
            st.error(f"Error setting up QA system: {str(e)}")
            return False

    def ask_question(self, question: str) -> Dict:
        """Ask a question and get a response with source information"""
        if not self.qa_chain:
            return {"error": "QA system not initialized. Please upload documents first."}
        try:
            result = self.qa_chain.invoke({
                "question": question,
                "chat_history": self.chat_history
            })
            
            sources = list({
                doc.metadata.get('source', 'Unknown source')
                for doc in result["source_documents"]
            })
            
            self.chat_history.append((question, result["answer"]))
            return {
                "answer": result["answer"],
                "sources": sources
            }
        except Exception as e:
            return {"error": f"Error processing question: {str(e)}"}

def main():
    st.set_page_config(
        page_title="Document QA Assistant",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("Document QA Assistant 🤖")

    # Initialize session state
    if 'manager' not in st.session_state:
        st.session_state.manager = DocumentManager()
    if 'api_key' not in st.session_state:
        st.session_state.api_key = ""
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Two-column layout
    col1, col2 = st.columns([1, 2])

    with col1:
        st.header("Configuration")
        api_key = st.text_input(
            "Enter your Google API Key",
            value=st.session_state.api_key,
            type="password"
        )
        if api_key:
            st.session_state.api_key = api_key
            os.environ["GOOGLE_API_KEY"] = api_key
            genai.configure(api_key=api_key)

        st.header("Upload Documents")
        uploaded_files = st.file_uploader(
            "Drag and drop your documents here",
            accept_multiple_files=True,
            type=['pdf', 'docx', 'pptx', 'ppt']
        )

        if uploaded_files and st.session_state.api_key:
            if st.button("Process Documents"):
                with st.spinner("Processing documents..."):
                    if st.session_state.manager.setup_qa_system(uploaded_files):
                        st.success("Documents processed successfully!")
                    else:
                        st.error("Failed to process documents.")

        if st.session_state.manager.processed_files:
            with st.expander("Processed Files"):
                for file in st.session_state.manager.processed_files:
                    st.text(f"✓ {file}")

    with col2:
        if st.session_state.manager.qa_chain:
            st.header("Ask Questions")
            
            # Display chat messages
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
                    if "sources" in message:
                        st.markdown("**Sources:**")
                        for source in message["sources"]:
                            st.markdown(f"- {source}")

            # Chat input
            if prompt := st.chat_input("Ask a question about your documents"):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.write(prompt)

                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = st.session_state.manager.ask_question(prompt)
                        if "error" in response:
                            st.error(response["error"])
                        else:
                            st.write(response["answer"])
                            if response["sources"]:
                                st.markdown("**Sources:**")
                                for source in response["sources"]:
                                    st.markdown(f"- {source}")
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": response["answer"],
                                "sources": response["sources"]
                            })
        else:
            st.info("Please upload documents and provide an API key to start.")

if __name__ == "__main__":
    main()
