import streamlit as st
import io
import contextlib
import sys
import re
from tools import initialize_agent


@contextlib.contextmanager
def capture_output():
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    try:
        sys.stdout = stdout_buffer
        sys.stderr = stderr_buffer
        yield stdout_buffer, stderr_buffer
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

def clean_output(text: str) -> str:
    ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
    return ansi_escape.sub('', text)

def handle_agent_chat(user_input, chat_memory, thinking_container, response_container, status_placeholder):
    """Handle agent mode chat with tools and verbose output"""
    try:
        # Initialize agent if needed
        if st.session_state.agent_executor is None:
            with st.spinner("🔄 Initializing AI Agent..."):
                agent_executor, llm = initialize_agent()
                st.session_state.agent_executor = agent_executor
                st.session_state.llm = llm
        
        if st.session_state.agent_executor is None:
            with response_container:
                st.error("Agent not initialized yet. Please wait.")
            status_placeholder.warning("Agent not ready")
            return "❌ Agent initialization failed"

        # Capture verbose output from agent
        with capture_output() as (stdout_buffer, stderr_buffer):
            chat_history = chat_memory.get_chat_history()
            agent_response = st.session_state.agent_executor.invoke({
                "input": user_input,
                "chat_history": chat_history
            })

        captured = stdout_buffer.getvalue().strip()
        captured_err = stderr_buffer.getvalue().strip()

        # Filter out tool output from thinking display
        thinking_lines = []
        for line in captured.splitlines():
            line = line.rstrip()
            if not line:
                continue
            # Only show agent framework messages, not tool output
            if any(phrase in line for phrase in [
                "Entering new AgentExecutor chain",
                "Invoking:",
                "Finished chain",
                "> Finished",
                "Action:",
                "Action Input:",
                "Observation:",
                "Thought:"
            ]):
                thinking_lines.append(line)

        # Display filtered thinking at TOP
        with thinking_container:
            thinking_container.empty()
            with st.expander("🧠 Agent Using Tools...", expanded=False):
                if thinking_lines:
                    for line in thinking_lines:
                        line= clean_output(line)
                        if "Entering new AgentExecutor chain" in line:
                            st.write( line)
                        elif "Invoking:" in line:
                            st.write(line)
                        elif "Finished" in line or line.startswith("> Finished"):
                            st.write(line)
                        elif "Action:" in line:
                            st.write(line)
                        elif "Observation:" in line:
                            st.write(line)
                        else:
                            st.text(line)
                else:
                    st.text("No agent thinking captured.")

                if captured_err:
                    st.markdown("**Errors:**")
                    st.code(captured_err, language="text")

        # Get final output and display
        final_output = agent_response.get("output", "") if isinstance(agent_response, dict) else str(agent_response)
        
        with response_container:
            st.markdown(final_output)

        status_placeholder.success("Done!")
        return final_output

    except Exception as e:
        error_msg = f"❌ Error: {e}"
        with response_container:
            st.error(error_msg)
        with thinking_container:
            thinking_container.empty()
            st.warning("Processing aborted.")
        status_placeholder.error("Failed")
        return error_msg