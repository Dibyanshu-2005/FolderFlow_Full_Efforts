import os
import tempfile
from typing import List, Dict, Any
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document
import google.generativeai as genai
from pptx import Presentation

class CustomPPTLoader:
    """Custom PowerPoint loader with robust text extraction"""
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> List[Document]:
        try:
            prs = Presentation(self.file_path)
            documents = []
            
            for slide_num, slide in enumerate(prs.slides, 1):
                text_content = []
                
                # Extract text from all possible shape types
                for shape in slide.shapes:
                    # Text from text boxes and other shapes with text
                    if hasattr(shape, "text") and shape.text.strip():
                        text_content.append(shape.text.strip())
                    
                    # Text from tables
                    if shape.has_table:
                        table_text = []
                        for row in shape.table.rows:
                            row_text = []
                            for cell in row.cells:
                                if cell.text.strip():
                                    row_text.append(cell.text.strip())
                            if row_text:
                                table_text.append(" | ".join(row_text))
                        if table_text:
                            text_content.append("\n".join(table_text))
                
                # Only create a document if there's actual content
                if text_content:
                    text = "\n\n".join(text_content)
                    metadata = {
                        "source": self.file_path,
                        "slide_number": slide_num
                    }
                    documents.append(Document(
                        page_content=text,
                        metadata=metadata
                    ))
            
            return documents
        except Exception as e:
            st.error(f"Error processing PowerPoint file: {str(e)}")
            return []

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
        
    def process_file(self, file_path: str, progress_bar) -> List[str]:
        """Process a single file and return its chunks"""
        try:
            if file_path.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
            elif file_path.endswith('.docx'):
                loader = Docx2txtLoader(file_path)
            elif file_path.endswith(('.pptx', '.ppt')):
                loader = CustomPPTLoader(file_path)
            else:
                return []
                
            documents = loader.load()
            progress_bar.progress(0.5)
            
            if not documents:
                st.warning(f"No content extracted from {os.path.basename(file_path)}")
                return []
                
            chunks = self.text_splitter.split_documents(documents)
            self.processed_files.append(os.path.basename(file_path))
            progress_bar.progress(1.0)
            
            return chunks
        except Exception as e:
            st.error(f"Error processing {os.path.basename(file_path)}: {str(e)}")
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
            
            if not all_chunks:
                st.error("No documents were successfully processed!")
                return False
            
            status_text.text("Initializing embeddings...")
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            
            status_text.text("Creating vector store...")
            vectorstore = Chroma.from_documents(
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
    st.set_page_config(page_title="Document QA Assistant", page_icon="📚")
    st.title("Document QA Assistant 🤖")

    # Initialize session state variables
    if 'manager' not in st.session_state:
        st.session_state.manager = None
    if 'system_ready' not in st.session_state:
        st.session_state.system_ready = False
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Configure Google API
    api_key = st.sidebar.text_input("Enter Google API Key", type="password")
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
        genai.configure(api_key=api_key)

    # File upload section
    st.sidebar.header("Upload Documents")
    uploaded_files = st.sidebar.file_uploader(
        "Drop your documents here",
        type=['pdf', 'docx', 'pptx', 'ppt'],
        accept_multiple_files=True,
        help="Supported formats: PDF, DOCX, PPTX, PPT"
    )

    # Initialize system button
    if uploaded_files and api_key and st.sidebar.button("Process Documents"):
        with st.spinner("Processing documents..."):
            manager = DocumentManager()
            if manager.setup_qa_system(uploaded_files):
                st.session_state.manager = manager
                st.session_state.system_ready = True
                st.sidebar.success("System initialized successfully!")
            else:
                st.sidebar.error("Failed to initialize the system.")

    # Main chat interface
    if st.session_state.system_ready:
        # Display processed files
        if st.session_state.manager.processed_files:
            with st.expander("Processed Files"):
                for file in st.session_state.manager.processed_files:
                    st.text(f"✓ {file}")

        st.markdown("### Ask questions about your documents")
        
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
                            unique_sources = {os.path.basename(source) for source in response["sources"]}
                            for source in unique_sources:
                                st.markdown(f"- {source}")
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response["answer"],
                            "sources": response["sources"]
                        })
    else:
        st.info("Please upload documents and enter your API key to begin.")

if __name__ == "__main__":
    main()
