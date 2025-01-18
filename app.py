import os
from typing import List, Dict, Any
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
import google.generativeai as genai
from pathlib import Path
import shutil
import nltk
import docx2txt

# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('punkt_tab')
    except Exception as e:
        st.warning(f"NLTK Download Warning: {str(e)}")

# Call the download function
download_nltk_data()

class DocumentManager:
    def __init__(self, upload_dir: str):
        self.upload_dir = upload_dir
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        self.processed_files = []
        self.qa_chain = None
        self.chat_history = []

    def process_file(self, file_path: str) -> List[str]:
        """Process a single file and return its chunks"""
        try:
            if file_path.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
            elif file_path.endswith('.docx'):
                # First try Docx2txtLoader, fallback to direct docx2txt
                try:
                    loader = Docx2txtLoader(file_path)
                except Exception as e:
                    st.warning(f"Falling back to direct docx2txt for {file_path}")
                    text = docx2txt.process(file_path)
                    return [{"page_content": text, "metadata": {"source": file_path}}]
            elif file_path.endswith(('.pptx', '.ppt')):
                loader = UnstructuredPowerPointLoader(file_path)
            else:
                st.error(f"Unsupported file type: {file_path}")
                return []
            
            documents = loader.load()
            chunks = self.text_splitter.split_documents(documents)
            self.processed_files.append(file_path)
            return chunks
        except Exception as e:
            st.error(f"Error processing {file_path}: {str(e)}")
            return []

    def setup_qa_system(self, api_key: str):
        """Initialize the QA system with processed documents"""
        try:
            # Configure API
            os.environ["GOOGLE_API_KEY"] = api_key
            genai.configure(api_key=api_key)

            all_chunks = []
            for root, _, files in os.walk(self.upload_dir):
                for file in files:
                    if file.endswith(('.pdf', '.docx', '.pptx', '.ppt')):
                        file_path = os.path.join(root, file)
                        chunks = self.process_file(file_path)
                        all_chunks.extend(chunks)

            if not all_chunks:
                st.error("No documents were successfully processed!")
                return False

            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            persist_directory = os.path.join(os.path.dirname(self.upload_dir), 'chroma_db')
            os.makedirs(persist_directory, exist_ok=True)

            vectorstore = Chroma.from_documents(
                documents=all_chunks,
                embedding=embeddings,
                persist_directory=persist_directory
            )

            self.qa_chain = ConversationalRetrievalChain.from_llm(
                ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7),
                vectorstore.as_retriever(),
                return_source_documents=True
            )
            return True
        except Exception as e:
            st.error(f"Error setting up QA system: {str(e)}")
            return False

    def ask_question(self, question: str) -> Dict:
        """Ask a question and get a response with source information"""
        if not self.qa_chain:
            return {"error": "QA system not initialized. Please set up the system first."}
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

def save_uploaded_files(uploaded_files):
    """Save uploaded files to a temporary directory"""
    # Create upload directory if it doesn't exist
    upload_dir = Path("uploaded_files")
    if upload_dir.exists():
        shutil.rmtree(upload_dir)
    upload_dir.mkdir(exist_ok=True)
    
    # Save uploaded files
    for uploaded_file in uploaded_files:
        file_path = upload_dir / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())
    
    return str(upload_dir)

def main():
    st.set_page_config(page_title="Document Assistant", page_icon="📚")
    st.title("Document Assistant 🤖")

    # Initialize session state variables
    if 'manager' not in st.session_state:
        st.session_state.manager = None
    if 'system_ready' not in st.session_state:
        st.session_state.system_ready = False
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # API Key input
    api_key = st.sidebar.text_input("Enter Google AI API Key", type="password")
    
    # File upload
    st.sidebar.header("Upload Documents")
    uploaded_files = st.sidebar.file_uploader(
        "Drag and drop your documents here",
        accept_multiple_files=True,
        type=['pdf', 'docx', 'pptx', 'ppt']
    )

    # Initialize button
    if st.sidebar.button("Initialize System"):
        if not api_key:
            st.sidebar.error("Please enter your API key.")
            return
        if not uploaded_files:
            st.sidebar.error("Please upload at least one document.")
            return

        with st.spinner("Setting up the document management system..."):
            # Save uploaded files and create manager
            upload_dir = save_uploaded_files(uploaded_files)
            manager = DocumentManager(upload_dir)
            
            if manager.setup_qa_system(api_key):
                st.session_state.manager = manager
                st.session_state.system_ready = True
                st.sidebar.success("System initialized successfully!")
            else:
                st.sidebar.error("Failed to initialize the system. Please check your API key and documents.")

    # Main chat interface
    if st.session_state.system_ready:
        # Display processed files
        if st.session_state.manager.processed_files:
            with st.expander("Processed Files"):
                for file in st.session_state.manager.processed_files:
                    st.text(f"✓ {os.path.basename(file)}")

        # Chat interface
        st.markdown("### Ask me anything about your documents!")
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                if "sources" in message:
                    st.markdown("**Sources:**")
                    unique_sources = {os.path.basename(source) for source in message["sources"]}
                    for source in unique_sources:
                        st.markdown(f"- {source}")

        # Chat input
        if prompt := st.chat_input("Your question"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.write(prompt)

            # Generate and display assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = st.session_state.manager.ask_question(prompt)
                    if "error" in response:
                        st.error(response["error"])
                    else:
                        st.write(response["answer"])
                        if response["sources"]:
                            st.markdown("**Sources:**")
                            unique_sources = {os.path.basename(source) for source in response["sources"]}
                            for source in unique_sources:
                                st.markdown(f"- {source}")
                        
                        # Add assistant response to chat history
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response["answer"],
                            "sources": response["sources"]
                        })
    else:
        st.info("Please enter your API key and upload documents using the sidebar controls.")

if __name__ == "__main__":
    main()
