import os
import time
import json
import requests
import shutil
import urllib.parse
import fitz
import io
import tempfile
import re
from typing import Optional, Type
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage
from langchain.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor
from ddgs import DDGS

#-----------PDF SUMMARIZATION------------
class PDFSummarizationInput(BaseModel):
    pdf_path: str = Field(description="Path to the PDF file or URL to download PDF for summarization")
    summary_length: Optional[str] = Field(default="medium", description="Length of summary: 'short', 'medium', or 'long'")


class PDFSummarizationTool(BaseTool):
    name: str = "pdf_summarization"
    description: str = "Summarize PDF documents. Can handle local files or download from URLs. Saves PDFs to local folder for future use."
    args_schema: Type[BaseModel] = PDFSummarizationInput
    
    class Config:
        arbitrary_types_allowed = True
        extra = "allow"
    
    def __init__(self):
        super().__init__()
        self.pdf_storage_folder = os.path.join(os.getcwd(), "saved_pdfs")
        os.makedirs(self.pdf_storage_folder, exist_ok=True)
        self.llm = ChatOllama(model="llama3.2", temperature=0.3, base_url="http://localhost:11434")
    
    def _run(self, pdf_path: str, summary_length: Optional[str] = "medium") -> str:
        try:
            local_pdf_path = self._handle_pdf_file(pdf_path)
            if not local_pdf_path:
                return "Error: Could not process PDF file"
            
            content = self._extract_pdf_content(local_pdf_path)
            if not content:
                return "Error: Could not extract text from PDF"
            
            summary = self._generate_summary(content, summary_length if summary_length else 'medium')
            return self._format_summary_response(summary, local_pdf_path, len(content))
            
        except Exception as e:
            return f"Error summarizing PDF: {str(e)}"
    
    def _handle_pdf_file(self, pdf_path: str) -> str:
        """Handle PDF file - download if URL, copy if local file, save to storage folder"""
        try:
            if pdf_path.startswith('http'):
                response = requests.get(pdf_path, timeout=15)
                response.raise_for_status()
                
                filename = os.path.basename(urllib.parse.urlparse(pdf_path).path)
                if not filename.endswith('.pdf'):
                    filename = f"downloaded_pdf_{int(time.time())}.pdf"
                
                local_path = os.path.join(self.pdf_storage_folder, filename)
                with open(local_path, 'wb') as f:
                    f.write(response.content)
                return local_path
                
            else:
                if os.path.exists(pdf_path):
                    abs_pdf_path = os.path.abspath(pdf_path)
                    abs_storage_folder = os.path.abspath(self.pdf_storage_folder)
                    
                    if abs_pdf_path.startswith(abs_storage_folder):
                        return pdf_path
                    
                    filename = os.path.basename(pdf_path)
                    local_path = os.path.join(self.pdf_storage_folder, filename)
                    
                    if os.path.exists(local_path):
                        name, ext = os.path.splitext(filename)
                        timestamp = int(time.time())
                        filename = f"{name}_{timestamp}{ext}"
                        local_path = os.path.join(self.pdf_storage_folder, filename)
                    
                    shutil.copy2(pdf_path, local_path)
                    return local_path
                else:
                    return ''
                    
        except Exception as e:
            return ''
    
    def _extract_pdf_content(self, pdf_path: str) -> str:
        """Extract text content from PDF"""
        try:
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            content = "\n\n".join([page.page_content for page in pages])
            return content
        except Exception as e:
            return ""
    
    def _generate_summary(self, content: str, summary_length: str) -> str:
        """Generate summary using LLM (no chunking)"""
        try:
            length_params = {
                "short": {"max_words": 150, "style": "brief bullet points"},
                "medium": {"max_words": 300, "style": "concise paragraphs"},
                "long": {"max_words": 500, "style": "detailed analysis"}
            }
            
            params = length_params.get(summary_length, length_params["medium"])
            return self._summarize_chunk(content, params)
                    
        except Exception as e:
            return f"Error generating summary: {str(e)}"
    
    def _summarize_chunk(self, text: str, params: dict) -> str:
        """Summarize a single chunk of text"""
        try:
            prompt = f"""
            Summarize the following text in {params['style']} format.
            Keep the summary under {params['max_words']} words.
            Focus on the main points, key findings, and important details.
            
            Text to summarize:
            {text}
            
            Summary:
            """
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return str(response.content).strip() 
            
        except Exception as e:
            return f"Error in chunk summarization: {str(e)}"
    
    def _format_summary_response(self, summary: str, pdf_path: str, content_length: int) -> str:
        """Format the final summary response"""
        filename = os.path.basename(pdf_path)
        
        response = f"""📄 PDF Summary: {filename}


        Summary:
        {summary}


        """
        
        return response.strip()

#-----------Web Search------------
class WebSearchTool(BaseTool):
    name: str = "web_search"
    description: str = "Search the web for current information, news, and general queries"
    
    def _run(self, query: str) -> str:
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=3))
                if not results:
                    return "No search results found."
                
                formatted_results = []
                for i, result in enumerate(results, 1):
                    formatted_results.append(f"{i}. {result.get('title', 'No title')}\n   {result.get('body', 'No description')}\n   URL: {result.get('href', 'No URL')}")
                
                return "\n\n".join(formatted_results)
        except Exception as e:
            return f"Search error: {str(e)}"



#-----------FORM FILLING------------
PDF_EXT = 'pdf'
# Add this new function to display the filled PDF

# Function to read PDF bytes
def get_pdf_bytes(filepath: str) -> bytes:
    with open(filepath, "rb") as f:
        pdf_bytes = f.read()
        return pdf_bytes
    
# Function to extract text from PDF
def extract_pdf_text(pdf_bytes: bytes) -> str:
    pdf_text = ''
    pdf_document = fitz.open(PDF_EXT, pdf_bytes)
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text_page = page.get_textpage()
        pdf_text += text_page.extractText()

    return pdf_text

# Function to extract form fields from PDF
def extract_pdf_fields(pdf_bytes: bytes) -> list[dict]:
    form_fields = []
    pdf_document = fitz.open(PDF_EXT, pdf_bytes)

    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        widget_list = page.widgets()
        if widget_list:
            for widget in widget_list:
                form_fields.append({
                    'name': widget.field_name,
                    'label': widget.field_label,
                    'type': widget.field_type_string,
                    'max_length': widget.text_maxlen
                })

    return form_fields

# Function to read PDF and extract text and fields
def read_pdf(filepath: str) -> tuple[str, list[dict]]:
    pdf_bytes = get_pdf_bytes(filepath)
    text = extract_pdf_text(pdf_bytes)
    fields = extract_pdf_fields(pdf_bytes)
    
    return text, fields

def read_txt(filepath: str) -> str:
    with open(filepath, 'r', encoding='utf-8') as file:
        contents = file.read()
        return contents

def fill_fields_prompt(pdf_text: str, fields: list[dict], source_info: str) -> str:
    return f"""
        You are an automated PDF forms filler.
        Your job is to fill the following form fields using the provided materials.
        Use the EXACT field "name" (not the label) as the key in your JSON response.
        
        Form fields with their names and labels:
        {json.dumps(fields, indent=2)}

        Materials:
        - Text extracted from the PDF form:
        <{pdf_text}>

        - Source info from user:
        #{source_info}#
        
        Output a JSON object with key-value pairs where:
        - key is the EXACT "name" from the field list above
        - value is the field value you determined from the materials
        
        Only include fields where you can determine a value. Skip buttons and fields you cannot fill.
        For CheckBox fields, use "Yes" or "No" as values.
    """

def call_openai(prompt: str, gpt_model: str = 'gpt-4o'):
    # Use ChatOllama for form filling
    form_llm = ChatOllama(model="llama3.2", temperature=0.1, base_url="http://localhost:11434")
    response = form_llm.invoke([HumanMessage(content=prompt)])

    response_data = response.content
    # return json.loads(response_data)

    # Look for JSON object pattern in the response
    response_str = str(response_data)
    json_match = re.search(r'\{.*\}', response_str, re.DOTALL)
    
    if json_match:
        json_str = json_match.group()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # If JSON parsing fails, try to clean up the string
            cleaned_json = json_str.replace('\n', '').replace('\t', '')
            return json.loads(cleaned_json)
    else:
        # If no JSON found, return empty dict
        print("No JSON found in response, returning empty dict")
        return {}

def fill_fields_with_gpt(pdf_text: str, fields: list[dict], source_info: str) -> list[dict]:
    prompt = fill_fields_prompt(pdf_text, fields, source_info)
    filled_fields_dict = call_openai(prompt)
    return filled_fields_dict

def fill_pdf_fields(pdf_bytes: bytes, field_values: dict) -> io.BytesIO:
    pdf_document = fitz.open(PDF_EXT, pdf_bytes)
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        widget_list = page.widgets()

        if widget_list:
            for widget in widget_list:
                field_name = widget.field_name
                if field_name in field_values:
                    widget.field_value = field_values[field_name]
                    widget.update()
                    
    output_stream = io.BytesIO()
    pdf_document.save(output_stream)
    output_stream.seek(0)

    return output_stream

def fill_pdf(fields_dict: dict, input_pdf: str, output_pdf: str):
    pdf_bytes = get_pdf_bytes(input_pdf)
    output_pdf_stream = fill_pdf_fields(pdf_bytes, fields_dict)
    
    with open(output_pdf, "wb") as f:
        f.write(output_pdf_stream.getvalue())

def fill_pdf_with_ai(input_pdf: str, output_pdf: str, source_file: str):
    print(f"Processing PDF: {input_pdf}")
        
    # Read PDF and source info
    pdf_text, pdf_fields = read_pdf(input_pdf)
    print(f"Extracted {len(pdf_fields)} form fields")
    
    if source_file and os.path.exists(source_file):
        source_info = read_txt(source_file)
        print(f"Read source info from: {source_file}")
    else:
        source_info = "No additional source information provided"
        print("No source file found, proceeding without additional info")
    
    # Use enhanced function
    filled_fields_dict = fill_fields_with_gpt(pdf_text, pdf_fields, source_info)
    
    print(f"Extracted field mappings: {filled_fields_dict}")
    
    if filled_fields_dict:
        # Fill the PDF
        fill_pdf(filled_fields_dict, input_pdf, output_pdf)
        print(f"Successfully created filled PDF: {output_pdf}")
    else:
        print("No field mappings extracted")

class FormFillingInput(BaseModel):
    pdf_path: str = Field(description="Path to the PDF file or URL to download PDF")
    source_path: Optional[str] = Field(default=None, description="Path to source information file or URL")

def display_filled_pdf(pdf_path: str) -> str:
    """Display filled PDF in Streamlit interface"""
    try:
        import streamlit as st
        
        if not os.path.exists(pdf_path):
            return f"❌ PDF file not found: {pdf_path}"
        
        # Create a container for the PDF display
        with st.container():
            st.markdown("### 📄 Filled PDF Form")
            
            # Read the PDF file
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()
            
            # Display PDF viewer
            st.download_button(
                label="📥 Download Filled PDF",
                data=pdf_bytes,
                file_name=os.path.basename(pdf_path),
                mime="application/pdf"
            )
            
            # Show PDF in browser (if supported)
            st.markdown(f"**File:** `{pdf_path}`")
            st.markdown(f"**Size:** {len(pdf_bytes):,} bytes")
            
            # PDF viewer using iframe (some browsers support this)
            import base64
            base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
            
            pdf_display = f'''
            <iframe src="data:application/pdf;base64,{base64_pdf}" 
                    width="100%" height="600" type="application/pdf">
                <p>Your browser does not support PDFs. 
                <a href="data:application/pdf;base64,{base64_pdf}">Download PDF</a></p>
            </iframe>
            '''
            
            st.markdown(pdf_display, unsafe_allow_html=True)
        
        return "✅ PDF displayed successfully"
        
    except Exception as e:
        return f"❌ Error displaying PDF: {str(e)}"

class FormFillingTool(BaseTool):
    name: str = "form_filling"
    description: str = "Fill PDF forms automatically using AI. Use when user wants to fill out a PDF form or complete form fields. Requires a PDF file path and optionally a source file with information to fill the form."
    args_schema: Type[BaseModel] = FormFillingInput  # Add this line like PDFExtractionTool

    def __init__(self):
        super().__init__()
    
    
        
    def _run(self, pdf_path: str, source_path: Optional[str] = None) -> str:
        """
        Process PDF form filling with proper input parameters.
        Args:
            pdf_path: Path to PDF file or URL
            source_path: Path to source information file or URL (optional)
        """
        try:
            # Validate inputs
            if not pdf_path:
                return "❌ PDF path is required"
            
            # Handle URL vs local file detection
            if pdf_path.lower().startswith(("http://", "https://")):
                return self._handle_url_form(pdf_path, source_path)
            else:
                return self._handle_local_form(pdf_path, source_path)
                
        except Exception as e:
            return f"❌ Error processing form: {str(e)}"

    def _handle_local_form(self, pdf_path: str, source_path: Optional[str] = None) -> str:
        """Handle local PDF form filling using your proven functions"""
        output_pdf = pdf_path.replace('.pdf', '_filled.pdf')
        return self._process_local_pdf(pdf_path, output_pdf, source_path or "")
    
    def _handle_url_form(self, pdf_url: str, source_path: Optional[str] = None) -> str:
        """Handle PDF form filling from URL"""
        # Download PDF from URL
        temp_pdf_path = self._download_pdf(pdf_url)
        if not temp_pdf_path:
            return f"❌ Failed to download PDF from: {pdf_url}"
        
        try:
            # Generate output filename
            output_pdf = f"downloaded_form_filled_{int(time.time())}.pdf"
            
            # Handle source info
            temp_source_file = ""
            if source_path:
                if source_path.lower().startswith(("http://", "https://")):
                    # Download source from URL
                    try:
                        response = requests.get(source_path, timeout=15)
                        response.raise_for_status()
                        temp_source_file = f"temp_source_{int(time.time())}.txt"
                        with open(temp_source_file, 'w', encoding='utf-8') as f:
                            f.write(response.text)
                    except Exception as e:
                        return f"❌ Failed to download source info: {str(e)}"
                else:
                    # Use local source file
                    temp_source_file = source_path
            
            result = self._process_local_pdf(temp_pdf_path, output_pdf, temp_source_file)
            return result
            
        finally:
            # Clean up temp files
            if os.path.exists(temp_pdf_path):
                os.unlink(temp_pdf_path)
            if temp_source_file and temp_source_file.startswith("temp_"):
                if os.path.exists(temp_source_file):
                    os.unlink(temp_source_file)

    def _download_pdf(self, url: str) -> str:
        """Download PDF from URL"""
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                temp_file.write(response.content)
                return temp_file.name
        except Exception as e:
            return ""

    def _process_local_pdf(self, input_pdf: str, output_pdf: str, source_file: str) -> str:
        """
        Process PDF using your proven functions
        """
        try:
            # Use your existing proven functions directly
            pdf_text, pdf_fields = read_pdf(input_pdf)
            
            if not pdf_fields:
                return f"❌ No form fields found in PDF: {input_pdf}"
            
            # Read source info
            if source_file and os.path.exists(source_file):
                source_info = read_txt(source_file)
                source_msg = f"Using source file: {source_file}"
            else:
                source_info = "No additional source information provided"
                source_msg = "No source file provided"
            
            # Use your proven AI function
            filled_fields_dict = fill_fields_with_gpt(pdf_text, pdf_fields, source_info)
            
            if not filled_fields_dict:
                return f"❌ No field mappings could be extracted. {source_msg}"
            
            # Fill and save PDF using your proven function
            fill_pdf(filled_fields_dict, input_pdf, output_pdf)

            try:
                display_result = display_filled_pdf(output_pdf)
            except Exception as display_error:
                display_result = f"PDF created but display failed: {display_error}"
            
            # Return detailed success message
            result = f"✅ Successfully filled PDF form: {output_pdf}\n"
            result += f"📊 Form fields found: {len(pdf_fields)}\n"
            result += f"📝 Fields filled: {len(filled_fields_dict)}\n"
            result += f"📄 {source_msg}\n"
            
            # Show sample of filled fields
            sample_fields = list(filled_fields_dict.keys())[:3]
            if sample_fields:
                result += f"🔧 Sample filled fields: {', '.join(sample_fields)}..."
            
            return result
            
        except Exception as e:
            return f"❌ Processing error: {str(e)}"
        




def initialize_agent():
    """Initialize tools and agent"""
    try:
        web_search = WebSearchTool()
        pdf_tool = PDFSummarizationTool()
        fill_form = FormFillingTool()
        tools = [web_search, pdf_tool, fill_form]
        
        llm = ChatOllama(
            model="llama3.2",
            temperature=0.1,
            base_url="http://localhost:11434"
        )
        
        system_prompt = """You are a helpful AI assistant with access to web search and PDF summarization tools.

You have access to the following tools:
1. web_search: Use this to search the web for current information, news, facts, or when you need up-to-date information
2. pdf_summarization: Use this to summarize PDF documents (either local paths or URLs). PDFs are automatically saved locally for future use.
3. form_filling: Use this to automatically fill PDF forms using AI. Use when user wants to fill out forms, complete applications, or populate PDF fields with information.

Guidelines:
- Use web_search when the user asks about current events, recent information, or general knowledge
- Use pdf_summarization when the user wants to summarize PDF content. You can specify summary length as 'short', 'medium', or 'long'
- Use fill_form when the user askes to fill in a pdf form using information source in the directory
- Always provide clear, helpful responses based on the information you gather
- When summarizing PDFs, the tool automatically saves them locally for future reference
- When filling forms, provide the PDF path and optionally a source file path with the information to use

Examples of when to use form_filling:
- "Fill out this application form"
- "Complete this PDF form using my information"
- "Populate the form fields in this document"
- "Auto-fill this form"

Think step by step about which tool(s) to use based on the user's question."""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ('placeholder', "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        
        agent = create_tool_calling_agent(llm, tools, prompt)
        
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=2,
            max_execution_time=120,
            early_stopping_method="generate"
        )
        
        return agent_executor, llm
    except Exception as e:
        return None, None
