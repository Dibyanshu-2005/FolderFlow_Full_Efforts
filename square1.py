import os
import tempfile
from typing import List, Dict, Any
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS  # Using FAISS instead of Chroma
from langchain.chains import ConversationalRetrievalChain
import google.generativeai as genai
import docx2txt
from pptx import Presentation
from langchain.schema import Document

class CustomPPTLoader:
    """Custom PowerPoint loader that doesn't rely on NLTK"""
    def __init__(self, file_path):
        self.file_path = file_path

    def load(self) -> List[Document]:
        prs = Presentation(self.file_path)
        documents = []
        
        for i, slide in enumerate(prs.slides, 1):
            text_content = []
            
            # Extract text from shapes in the slide
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text_content.append(shape.text)
            
            # Only create a document if there's actual content
            if text_content:
                text = "\n".join(text_content)
                metadata = {
                    "source": self.file_path,
                    "slide_number": i
                }
                documents.append(Document(
                    page_content=text,
                    metadata=metadata
                ))
        
        return documents

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
        self.temp_dir = tempfile.mkdtemp()
        
    def process_file(self, file_path: str, progress_bar=None) -> List[str]:
        """Process a single file and return its chunks"""
        try:
            print(f"Processing: {file_path}")
            
            if file_path.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
            elif file_path.endswith('.docx'):
                # Custom handling for Word documents
                text = docx2txt.process(file_path)
                documents = [Document(
                    page_content=text,
                    metadata={"source": file_path}
                )]
                return self.text_splitter.split_documents(documents)
            elif file_path.endswith(('.pptx', '.ppt')):
                loader = CustomPPTLoader(file_path)
            else:
                return []
                
            if progress_bar:
                progress_bar.progress(0.3)
                
            documents = loader.load()
            if progress_bar:
                progress_bar.progress(0.6)
            
            chunks = self.text_splitter.split_documents(documents)
            self.processed_files.append(os.path.basename(file_path))
            
            if progress_bar:
                progress_bar.progress(1.0)
            
            return chunks
        except Exception as e:
            st.error(f"Error processing {os.path.basename(file_path)}: {str(e)}")
            print(f"Detailed error: {str(e)}")  # For debugging
            return []

    def setup_qa_system(self, uploaded_files):
        """Initialize the QA system with uploaded documents"""
        try:
            all_chunks = []
            
            # Create progress containers for each file
            progress_bars = {file.name: st.progress(0) for file in uploaded_files}
            status_text = st.empty()
            
            for uploaded_file in uploaded_files:
                status_text.text(f"Processing {uploaded_file.name}...")
                
                # Save uploaded file to temporary directory
                temp_path = os.path.join(self.temp_dir, uploaded_file.name)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                # Process the file
                chunks = self.process_file(temp_path, progress_bars[uploaded_file.name])
                all_chunks.extend(chunks)
                
                # Clean up temporary file
                os.remove(temp_path)
            
            if not all_chunks:
                st.error("No documents were successfully processed!")
                return False
            
            status_text.text("Initializing embeddings...")
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            
            status_text.text("Creating vector store...")
            vectorstore = FAISS.from_documents(
                documents=all_chunks,
                embedding=embeddings
            )
            
            status_text.text("Setting up QA chain...")
            self.qa_chain = ConversationalRetrievalChain.from_llm(
                ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7),
                vectorstore.as_retriever(),
                return_source_documents=True
            )
            
            status_text.text("Setup complete!")
            return True
            
        except Exception as e:
            st.error(f"Error setting up QA system: {str(e)}")
            print(f"Detailed error: {str(e)}")  # For debugging
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

# Rest of your main() function remains the same...
