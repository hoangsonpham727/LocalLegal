import os
import pandas as pd
import streamlit as st
import requests
from bs4 import BeautifulSoup
from typing import Any, Dict, List
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_core.messages import HumanMessage
from langchain_community.llms import Ollama
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Configuration
PERSIST_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2"

# Web scraping headers
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}

PROMPT = PromptTemplate(
    template="""Use the following context to answer the question. If you don't know the answer, say you don't know.
Context: {context}
Question: {question}""",
    input_variables=["context", "question"],
)

def process_excel_data(excel_path: str) -> List[Document]:
    """Process Excel file and extract business-related information"""
    try:
        # Read the Excel file
        df = pd.read_excel(excel_path)
        
        # Convert all column names to lowercase for easier matching
        df.columns = df.columns.str.lower()
        
        # Define business-related keywords to filter relevant rows
        business_keywords = [
            'business', 'company', 'corporation', 'enterprise', 'firm', 'organization',
            'owner', 'director', 'manager', 'entrepreneur', 'proprietor', 'partnership',
            'commerce', 'trade', 'industry', 'commercial', 'economic', 'financial',
            'registration', 'license', 'permit', 'compliance', 'regulation',
            'tax', 'abn', 'acn', 'gst', 'employment', 'workplace'
        ]
        
        documents = []


        # Process each row
        for index, row in df.iterrows():
            # Convert row to string for keyword searching
            row_text = ' '.join([str(val) for val in row.values if pd.notna(val)]).lower()
            
            # Check if row contains business-related keywords
            if any(keyword in row_text for keyword in business_keywords):
                # Create a structured text representation of the row
                content_parts = []
                
                for col, value in row.items():
                    if pd.notna(value) and str(value).strip():
                        content_parts.append(f"{col}: {value}")
                
                if content_parts:
                    content = "\n".join(content_parts)
                    
                    # Create document with metadata
                    doc = Document(
                        page_content=content,
                        metadata={
                            "source": f"Excel: {os.path.basename(excel_path)}",
                            "row": index,
                            "type": "excel",
                            "category": "business_legislation"
                        }
                    )
                    documents.append(doc)
        return documents
        
    except Exception as e:
        st.error(f"Error processing Excel file {excel_path}: {e}")
        return []

def scrape_website(url: str) -> str:
    """Scrapes text from all <p> tags on a given URL."""
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract text from multiple elements for better coverage
        text_elements = []
        
        # Get paragraphs
        for p in soup.find_all('p'):
            text_elements.append(p.get_text().strip())
        
        # Get headings
        for heading in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            text_elements.append(heading.get_text().strip())
        
        # Get list items
        for li in soup.find_all('li'):
            text_elements.append(li.get_text().strip())
        
        # Join all text elements
        text = ' '.join([t for t in text_elements if t])
        
        return text if text.strip() else ""
        
    except Exception as e:
        st.error(f"Error scraping {url}: {e}")
        return ""

def _build_vector_db_from_sources():
    """Build vector DB from multiple sources: PDFs and web scraping"""
    
    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    
    all_chunks = []
    sources_processed = []
    
    # 1. Process PDFs from saved_pdfs directory
    pdf_dir = os.path.join(os.path.dirname(__file__), "datasets")
    if os.path.exists(pdf_dir):
        pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
        
        for pdf_file in pdf_files:
            try:
                pdf_path = os.path.join(pdf_dir, pdf_file)
                loader = PyPDFLoader(pdf_path)
                pages = loader.load()
                
                for page_num, page in enumerate(pages):
                    chunks = text_splitter.split_text(page.page_content)
                    for chunk_num, chunk in enumerate(chunks):
                        all_chunks.append(Document(
                            page_content=chunk,
                            metadata={
                                "source": f"PDF: {pdf_file}",
                                "page": page_num,
                                "chunk": chunk_num,
                                "type": "pdf"
                            }
                        ))
                
                sources_processed.append(f"PDF: {pdf_file}")
                
            except Exception as e:
                st.warning(f"Error processing PDF {pdf_file}: {e}")
    
    # 2. Process web sources (you can add more URLs here)
    web_sources = [
        "https://www.melbourne.vic.gov.au/portable-advertising-board-permits",
       "https://www.ato.gov.au/businesses-and-organisations/income-deductions-and-concessions/income-and-deductions-for-business/deductions"
    ]
    for url in web_sources:
        try:
            st.info(f"Scraping: {url}")
            raw_text = scrape_website(url)
            
            if raw_text:
                chunks = text_splitter.split_text(raw_text)
                for chunk_num, chunk in enumerate(chunks):
                    all_chunks.append(Document(
                        page_content=chunk,
                        metadata={
                            "source": url,
                            "chunk": chunk_num,
                            "type": "web"
                        }
                    ))
                
                sources_processed.append(f"Web: {url}")
            
        except Exception as e:
            st.warning(f"Error processing URL {url}: {e}")
    #3. Process Excel files from directory
    if os.path.exists(pdf_dir):  # Same directory as PDFs

        excel_files = [f for f in os.listdir(pdf_dir) 
                      if f.lower().endswith(('.xlsx', '.xls')) and 'inforce_commonwealth_legislation' in f.lower()]
        
        for excel_file in excel_files:
            try:
                excel_path = os.path.join(pdf_dir, excel_file)
                st.info(f"Processing Excel file: {excel_file}")
                
                excel_docs = process_excel_data(excel_path)
                
                # Split large Excel content into chunks if needed
                for doc in excel_docs:
                    if len(doc.page_content) > 1000:
                        chunks = text_splitter.split_text(doc.page_content)
                        for chunk_num, chunk in enumerate(chunks):
                            all_chunks.append(Document(
                                page_content=chunk,
                                metadata={
                                    **doc.metadata,
                                    "chunk": chunk_num
                                }
                            ))
                    else:
                        all_chunks.append(doc)
                
                sources_processed.append(f"Excel: {excel_file} ({len(excel_docs)} business records)")
                
            except Exception as e:
                st.warning(f"Error processing Excel file {excel_file}: {e}")
    
    if not all_chunks:
        return False, "No content found from any source"
    
    try:
        # Create vector database
        embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        os.makedirs(PERSIST_DIR, exist_ok=True)
        
        # Remove existing vector DB if it exists
        if os.path.exists(PERSIST_DIR):
            import shutil
            shutil.rmtree(PERSIST_DIR)
        
        chroma = Chroma.from_documents(
            documents=all_chunks,
            embedding=embeddings,
            persist_directory=PERSIST_DIR
        )
        chroma.persist()
        
        return True, f"Successfully created vector DB with {len(all_chunks)} chunks from {len(sources_processed)} sources: {', '.join(sources_processed)}"
        
    except Exception as e:
        return False, f"Error creating vector DB: {str(e)}"

def _init_rag() -> Dict[str, Any]:
    """Initialize or load RAG components and store them in session_state."""
    if "rag" in st.session_state:
        return st.session_state.rag

    # Check if vector DB exists
    if not os.path.exists(PERSIST_DIR):
        st.info("🔄 Vector database not found. Building from PDFs and web sources...")
        
        with st.spinner("Building vector database (this may take a few minutes)..."):
            success, message = _build_vector_db_from_sources()
        
        if success:
            st.success(f"✅ {message}")
        else:
            error_msg = f"❌ Failed to create vector DB: {message}"
            st.error(error_msg)
            st.session_state.rag = {"error": error_msg}
            return st.session_state.rag

    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        vector_db = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
        retriever = vector_db.as_retriever(search_kwargs={"k": 10})  # Retrieve top 10 relevant chunks

        llm = Ollama(model=LLM_MODEL)
        
        # Test LLM connection
        try:
            _ = llm.invoke("Hello")
        except Exception as e:
            raise Exception(f"LLM connection failed: {e}. Please ensure Ollama is running with model '{LLM_MODEL}'")

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT},
        )

        st.session_state.rag = {
            "embeddings": embeddings,
            "vector_db": vector_db,
            "retriever": retriever,
            "llm": llm,
            "qa_chain": qa_chain,
            "error": None,
        }
        
        return st.session_state.rag
        
    except Exception as e:
        error_msg = f"RAG initialization error: {str(e)}"
        st.session_state.rag = {"error": error_msg}
        return st.session_state.rag

def handle_rag_chat(user_input: str, chat_memory, thinking_container, response_container, status_placeholder) -> str:
    """Handle RAG-based chat with web scraping and PDF sources"""
    try:
        status_placeholder.info("🔎 Retrieving relevant context...")
        rag = _init_rag()
        
        if rag.get("error"):
            with response_container:
                st.error(rag["error"])
                st.info("💡 The system will automatically scrape web content and use any PDFs in saved_pdfs folder")
            status_placeholder.error("Failed")
            return rag["error"]

        qa = rag["qa_chain"]

        # Show thinking info
        with thinking_container:
            with st.expander("🧠 RAG Process (Web + PDF)", expanded=False):
                st.text("Searching vector database (web scraped content + PDFs) and generating answer...")

        # Run retrieval + answer
        result = qa({"query": user_input})
        answer = result.get("result", str(result))

        # Display answer and sources
        with response_container:
            st.markdown(answer)

            sources = result.get("source_documents", [])
            if sources:
                with st.expander("📚 Source Documents", expanded=False):
                    for i, doc in enumerate(sources, 1):
                        content = doc.page_content
                        metadata = doc.metadata
                        source_type = metadata.get('type', 'unknown')
                        source_name = metadata.get('source', 'Unknown')

                        
                        # Display source with type indicator
                        if source_type == 'pdf':
                            st.markdown(f"**📄 Source {i}** - {source_name}")
                        elif source_type == 'web':
                            st.markdown(f"**🌐 Source {i}** - {source_name}")
                        else:
                            st.markdown(f"**📖 Source {i}** - {source_name}")
                        
                        st.divider()

        status_placeholder.success("Done")
        return answer

    except Exception as e:
        error_msg = f"❌ RAG error: {e}"
        with response_container:
            st.error(error_msg)
        status_placeholder.error("Failed")
        return error_msg

# Function to manually rebuild vector DB (can be called from UI)
def rebuild_vector_db():
    """Manually rebuild the vector database"""
    if os.path.exists(PERSIST_DIR):
        import shutil
        shutil.rmtree(PERSIST_DIR)
    
    # Clear session state
    if "rag" in st.session_state:
        del st.session_state.rag
    
    return _build_vector_db_from_sources()