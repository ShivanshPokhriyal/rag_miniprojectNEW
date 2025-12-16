import streamlit as st
import requests

QUERY_URL = "http://127.0.0.1:8000/query"
UPLOAD_URL = "http://127.0.0.1:8000/upload"
RESET_URL = "http://127.0.0.1:8000/reset"



st.set_page_config(page_title="RAG Chatbot", layout="centered")
st.title("📄 RAG PDF Chatbot")

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar with conversation controls
with st.sidebar:
    st.header("Conversation Settings")
    
    with st.sidebar:
        st.header("Session Controls")

        st.caption(
        "Start a new session by clearing all uploaded documents "
        "and resetting the knowledge base."
    )

        if st.button("🆕 Start New Session"):
            try:
                requests.post(RESET_URL, timeout=10)
                st.session_state.messages = []
                st.success("New session started")
                st.rerun()
            except requests.exceptions.RequestException:
                st.error("Failed to reset session")

        st.divider()

        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

    
    # Display message count
    st.metric("Messages", len(st.session_state.messages))
    
    st.divider()
    st.caption("This chatbot remembers your conversation context")
    st.subheader("📤 Upload PDF for RAG")

uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"]
)

if uploaded_file is not None:
    files = {
        "file": (uploaded_file.name, uploaded_file, "application/pdf")
    }

    with st.spinner("Uploading & indexing document..."):
        try:
            response = requests.post(
                UPLOAD_URL,
                files=files,
                timeout=60
            )

            if response.status_code == 200:
                st.success("✅ Document uploaded and indexed!")
            else:
                st.error(f"❌ Upload failed: {response.text}")

        except requests.exceptions.RequestException as e:
            st.error(f"❌ Connection error: {str(e)}")

st.divider()


# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
query = st.chat_input("Ask a question from your PDFs...")

if query:
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": query})
    
    with st.chat_message("user"):
        st.markdown(query)

    # Get assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Send query with conversation history
                response = requests.post(
                    QUERY_URL,
                    json={
                        "question": query,
                        "history": st.session_state.messages[:-1]  # Exclude current message
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    answer = response.json()["answer"]
                else:
                    answer = f"Error: {response.status_code} - {response.text}"
                    
            except requests.exceptions.RequestException as e:
                answer = f"Connection error: {str(e)}"

        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})