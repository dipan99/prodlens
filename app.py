from langchain_core.messages import HumanMessage, AIMessage
from graph import ProdLensQueryEngine
from logger import Logging
import streamlit as st
import time

# Page configuration
st.set_page_config(
    page_title="ProdLens Chat",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f5f5f5;
        border-left: 4px solid #4caf50;
    }
    .message-header {
        font-weight: bold;
        margin-bottom: 0.5rem;
        color: #333;
    }
    .metadata {
        font-size: 0.8rem;
        color: #666;
        margin-top: 0.5rem;
        padding-top: 0.5rem;
        border-top: 1px solid #ddd;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

Logging.logDebug("Initializing session state")
if "engine" not in st.session_state:
    st.session_state.engine = ProdLensQueryEngine()
    st.session_state.thread_id = st.session_state.engine.thread_id

if "messages" not in st.session_state:
    st.session_state.messages = []

if "show_metadata" not in st.session_state:
    st.session_state.show_metadata = False


def display_message(role: str, content: str, metadata: dict = None):
    """Display a chat message with optional metadata."""
    message_class = "user-message" if role == "user" else "assistant-message"
    icon = "👤" if role == "user" else "🤖"
    
    st.markdown(f"""
    <div class="chat-message {message_class}">
        <div class="message-header">{icon} {role.capitalize()}</div>
        <div>{content}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Show metadata if enabled and available
    if st.session_state.show_metadata and metadata:
        with st.expander("🔍 Query Details", expanded=False):
            if metadata.get("standalone_query"):
                st.markdown(f"**Standalone Query:** {metadata['standalone_query']}")
            if metadata.get("route"):
                st.markdown(f"**Route:** {metadata['route'].upper()}")
            if metadata.get("reasoning"):
                st.markdown(f"**Reasoning:** {metadata['reasoning']}")
            if metadata.get("sql_result"):
                st.code(metadata["sql_result"], language="sql")


def clear_chat():
    """Clear chat history and start new conversation."""
    st.session_state.messages = []
    st.session_state.engine.new_conversation()
    st.session_state.thread_id = st.session_state.engine.thread_id


# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    
    # Thread ID display
    st.info(f"**Thread ID:** `{st.session_state.thread_id[:8]}...`")
    
    # Show metadata toggle
    st.session_state.show_metadata = st.checkbox(
        "Show Query Metadata",
        value=st.session_state.show_metadata,
        help="Display routing decisions and SQL queries"
    )
    
    st.divider()
    
    # Actions
    st.subheader("Actions")
    
    if st.button("🆕 New Conversation", use_container_width=True):
        clear_chat()
        st.rerun()
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    
    # Statistics
    st.subheader("📊 Statistics")
    st.metric("Total Messages", len(st.session_state.messages))
    
    user_msgs = len([m for m in st.session_state.messages if m["role"] == "user"])
    st.metric("User Queries", user_msgs)
    
    st.divider()
    
    # Help
    st.subheader("💡 Tips")
    st.markdown("""
    **Sample Queries:**
    - Show me gaming monitors under $500
    - What is refresh rate?
    - Compare mechanical keyboards
    - Tell me about the SteelSeries ones
    - What's the best mouse for gaming?
    """)
    
    st.divider()
    
    # About
    with st.expander("ℹ️ About ProdLens"):
        st.markdown("""
        **ProdLens** is an intelligent electronics assistant powered by:
        - **Text2SQL**: For querying product specifications
        - **RAG**: For explaining technical concepts
        - **LangGraph**: For intelligent query routing
        
        Ask questions naturally and the system will route them appropriately!
        """)


# Main chat interface
st.title("🔍 ProdLens Chat Assistant")
st.markdown("Ask me anything about electronics products!")

# Display chat history
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        display_message(
            message["role"],
            message["content"],
            message.get("metadata")
        )

# Chat input
user_input = st.chat_input("Ask about products or technical specs...")

if user_input:
    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
        "metadata": None
    })
    
    # Display user message immediately
    with chat_container:
        display_message("user", user_input)
    
    # Show loading spinner
    with st.spinner("🤔 Thinking..."):
        try:
            # Query the engine
            result = st.session_state.engine.query(
                user_input,
                thread_id=st.session_state.thread_id
            )
            
            # Extract response
            assistant_response = result.get("final_answer", "I couldn't process that query.")
            
            # Prepare metadata
            metadata = {
                "standalone_query": result.get("standalone_query"),
                "route": result.get("route"),
                "reasoning": result.get("reasoning"),
                "sql_result": result.get("sql_result") if result.get("route") == "text2sql" else None,
                "error": result.get("error")
            }
            
            # Add assistant message
            st.session_state.messages.append({
                "role": "assistant",
                "content": assistant_response,
                "metadata": metadata
            })
            
        except Exception as e:
            Logging.logError(str(e))
            st.session_state.messages.append({
                "role": "assistant",
                "content": str(e),
                "metadata": {"error": str(e)}
            })
    
    # Rerun to display assistant message
    st.rerun()


# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.8rem;'>
    Powered by LangGraph | OpenAI | PostgreSQL
</div>
""", unsafe_allow_html=True)