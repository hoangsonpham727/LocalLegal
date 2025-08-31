# Local Legal Assistant 🏛️

An intelligent AI-powered legal assistant that helps with PDF form filling, document summarization, web research, and RAG-based legal information retrieval. Built with Streamlit and LangChain.

## 🌟 Features

- **📄 PDF Form Filling**: Automatically fill PDF forms using AI with source information
- **📋 PDF Summarization**: Generate summaries of legal documents with customizable length
- **🔍 Web Search**: Real-time web search for current legal information
- **🧠 RAG Chat**: Retrieval-Augmented Generation using legal documents and web sources
- **🎯 Agent Mode**: Intelligent tool selection and chaining for complex tasks
- **📊 Multi-Source Knowledge**: Combines PDFs, Excel files, and web content

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **AI/ML**: LangChain, Ollama (Llama 3.2)
- **Document Processing**: PyMuPDF, PyPDFLoader
- **Vector Database**: Chroma
- **Embeddings**: HuggingFace Transformers
- **Web Scraping**: BeautifulSoup, DuckDuckGo Search

## 📋 Prerequisites

- Python 3.8+
- Ollama running locally with Llama 3.2 model
- Internet connection for web search and document downloads

## 🚀 Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd SnakeAI/Agent
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up Ollama**
```bash
# Install Ollama (if not already installed)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull the required model
ollama pull llama3.2
```

4. **Create necessary directories**
```bash
mkdir -p saved_pdfs datasets
```

## 📁 Project Structure

```
Agent/
├── app.py                          # Main Streamlit application
├── tools.py                        # AI tools (PDF, search, forms)
├── agent_chat.py                   # Agent mode chat handler
├── rag_chat.py                     # RAG-based chat handler
├── normal_chat.py                  # Simple chat handler
├── datasets/                       # Legal documents and data
│   ├── *.pdf                      # Legal PDF documents
│   └── *.xlsx                     # Commonwealth legislation data
├── saved_pdfs/                     # Uploaded and processed PDFs
├── img/                           # Application assets
└── __pycache__/                   # Python cache files
```

## 🏃‍♂️ Usage

1. **Start the application**
```bash
streamlit run app.py
```

2. **Access the interface**
   - Open your browser to `http://localhost:8501`

3. **Choose your mode**
   - **Agent Mode**: AI automatically selects and uses appropriate tools
   - **RAG Mode**: Query knowledge base of legal documents

### 📄 PDF Form Filling

```python
# Example usage
"Fill out the business application form using my business information"
"Complete the PDF form at /path/to/form.pdf using data from business-info.txt"
```

### 📋 Document Summarization

```python
# Example prompts
"Summarize this legal document with a medium summary"
"Give me a brief overview of the Business Licensing Act"
```

### 🔍 Web Research

```python
# Example queries
"Search for recent changes in business licensing requirements"
"What are the current tax obligations for new businesses?"
```

## ⚙️ Configuration

### Ollama Settings
```python
# In tools.py and rag_chat.py
LLM_MODEL = "llama3.2"
BASE_URL = "http://localhost:11434"
TEMPERATURE = 0.1
```

### Vector Database
```python
# In rag_chat.py
PERSIST_DIR = "../chroma_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
```

### PDF Processing
```python
# In tools.py
PDF_STORAGE = "saved_pdfs"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
```

## 📊 Data Sources

The system automatically processes:

1. **PDFs**: Legal documents in `datasets/` folder
2. **Excel Files**: Commonwealth legislation data
3. **Web Sources**: Melbourne city council permits
4. **User Uploads**: PDFs uploaded via the interface

## 🔧 Advanced Features

### Custom Tool Integration


### RAG Knowledge Base

The system automatically builds a vector database from:
- PDF documents in `datasets/`
- Excel files with business legislation
- Web-scraped content from legal sources

### Form Field Detection

Automatically detects and fills PDF form fields:
- Text fields
- Checkboxes
- Radio buttons
- Dropdown menus

## 🐛 Troubleshooting

### Common Issues

1. **Ollama Connection Error**
```bash
# Check if Ollama is running
ollama list
# Restart if needed
ollama serve
```

2. **Vector Database Issues**
```bash
# Clear and rebuild
rm -rf chroma_db/
# Restart the app to rebuild
```

3. **PDF Processing Errors**
   - Ensure PDFs are not password-protected
   - Check file permissions
   - Verify file paths are correct

4. **Memory Issues**
   - Reduce chunk size in configuration
   - Process fewer documents at once
   - Increase system memory allocation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Verify all prerequisites are met


**Local Legal Assistant** - Making legal document processing intelligent and efficient! 🏛️✨