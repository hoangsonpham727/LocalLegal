import streamlit as st
import sys
import os
from typing import List
from langchain_core.messages import HumanMessage, AIMessage
from langchain.schema import BaseMessage
from functools import wraps
import threading

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our modules
from agent_chat import handle_agent_chat
from rag_chat import handle_rag_chat, _init_rag

# Page config
st.set_page_config(
    page_title="Local Legal Assistant",
    page_icon="img/local-3.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'agent_executor' not in st.session_state:
    st.session_state.agent_executor = None
if 'llm' not in st.session_state:
    st.session_state.llm = None
if 'tools_initialized' not in st.session_state:
    st.session_state.tools_initialized = False
if 'agent_mode' not in st.session_state:
    st.session_state.agent_mode = False
try:
    _init_rag()
except Exception:
    pass

# Chat Memory Class
class StreamlitChatMemory:
    def __init__(self):
        if 'messages' not in st.session_state:
            st.session_state.messages = []
    
    def add_user_message(self, message: str):
        st.session_state.messages.append(HumanMessage(content=message))
        self._trim_messages()
    
    def add_ai_message(self, message: str):
        st.session_state.messages.append(AIMessage(content=message))
        self._trim_messages()
    
    def get_chat_history(self) -> List[BaseMessage]:
        return st.session_state.messages
    
    def _trim_messages(self):
        if len(st.session_state.messages) > 20:
            st.session_state.messages = st.session_state.messages[-20:]
    
    def clear(self):
        st.session_state.messages = []


# Timeout implementation
class TimeoutError(Exception):
    pass


def streamlit_timeout(seconds=30):
    """Timeout decorator that works with Streamlit"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = [None]
            exception = []
            
            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    exception.append(e)
            
            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(seconds)
            
            if thread.is_alive():
                raise TimeoutError(f"Function {func.__name__} timed out after {seconds} seconds")
            
            if exception[0]:
                raise exception[0]
            
            return result[0]
        
        return wrapper
    return decorator


# Initialize memory
chat_memory = StreamlitChatMemory()
LOGO_PATH = 'img/local-3.png'

def main():
    st.title("Local Legal")
    st.markdown("Your intelligent assistant with web search and form filling capabilities")
    
    # Sidebar
    with st.sidebar:
        st.image(LOGO_PATH, width=40, use_container_width=False)    
        
        # Agent mode toggle
        agent_toggle = st.checkbox("Agent Mode", value=st.session_state.agent_mode,
                                   help="When enabled the agent will use tools (web_search, pdf_summarization). Default is simple ask & answer.")
        st.session_state.agent_mode = bool(agent_toggle)
        
        # PDF Upload
        
        uploaded_file = st.file_uploader("", type="pdf")
        
        if uploaded_file is not None:
            # Save uploaded file
            pdf_dir = "saved_pdfs"
            os.makedirs(pdf_dir, exist_ok=True)
            pdf_path = os.path.join(pdf_dir, uploaded_file.name)
            
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.write(f"PDF saved: {uploaded_file.name}")
            
            # Summary length selector
            summary_length = st.selectbox(
                "Summary Length:",
                ["short", "medium", "long"],
                index=1
            )
            
            # Auto-suggest summarization
            if st.button("Summarize This PDF"):
                st.session_state.auto_question = f"Please summarize the PDF file: {pdf_path} with {summary_length} summary"
        # Clear chat button
        if st.button("Clear Chat History"):
            chat_memory.clear()
            st.session_state.chat_history = []
            st.rerun()
        
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"], avatar='img/local-3.png'):
            st.markdown(message["content"])
    
    # Handle auto-questions from sidebar
    if hasattr(st.session_state, 'auto_question'):
        user_input = st.session_state.auto_question
        delattr(st.session_state, 'auto_question')
    else:
        user_input = st.chat_input("Ask me anything...")
    
    if user_input:
        # Add user message to chat and display immediately
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar='img/4.png'):
            st.markdown(user_input)

        # ChatGPT-like layout: thinking above, answer below
        with st.chat_message("assistant", avatar='img/local-3.png'):
            # Show processing status at the top
            status_placeholder = st.empty()
            status_placeholder.write("🤔 Processing..." if st.session_state.agent_mode else "🤖 Querying LLM...")
            
            thinking_container = st.container()   # TOP: agent thinking / tool calls
            response_container = st.container()   # BOTTOM: final answer


            try:
                # Add to memory
                chat_memory.add_user_message(user_input)

                if st.session_state.agent_mode:
                    # Agent mode with tools
                    final_output = handle_agent_chat(user_input, chat_memory, thinking_container, response_container, status_placeholder)
                else:
                    # Simple LLM mode
                    final_output = handle_rag_chat(user_input, chat_memory, thinking_container, response_container, status_placeholder)

                # Add AI response to memory and chat history
                chat_memory.add_ai_message(final_output)
                st.session_state.chat_history.append({"role": "assistant", "content": final_output})

            except Exception as e:
                # Handle any uncaught exceptions
                error_msg = f"❌ Error: {str(e)}"
                with response_container:
                    st.error(error_msg)
                with thinking_container:
                    thinking_container.empty()
                    st.warning("Processing aborted.")
                status_placeholder.error("Failed")
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})

            finally:
                # Clear status placeholder
                try:
                    status_placeholder.empty()
                except:
                    pass


if __name__ == "__main__":
    main()