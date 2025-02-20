from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

from dotenv import load_dotenv
load_dotenv()
# Langsmith Tracking
os.environ["LANGCHAIN_API_KEY1"] = os.getenv("LANGCHAIN_API_KEY1")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Medical Chat Bot"

# load the data
loader = PyPDFLoader('data.pdf')
docs = loader.load()

# dividing the data into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=250, chunk_overlap=50)
final_documents = text_splitter.split_documents(docs)


# Emdedding
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# embeddings = OllamaEmbeddings(model_name="mistral")
doc_result = embeddings.embed_documents("data.pdf")

# vectors
db = FAISS.from_documents(docs, embeddings)


# storing data in vectordb
vectorstore = FAISS.from_documents(
    documents=final_documents, embedding=embeddings)
retriever = vectorstore.as_retriever()


# proving llm model
llm = Ollama(model="mistral",streaming=True)

# giving prompt
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# creating a rag chain
retriever = vectorstore.as_retriever(
    search_type="similarity", search_kwargs={"k": 2})
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# adding chat history

store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

import streamlit as st

# Title of the app
st.title("Medical Q&A Chatbot With Ollama")

# Display a prompt for user input
st.write("Go ahead and ask any medical question.")

# Create a chat history container
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Take user input (chat input box)
user_input = st.chat_input("You:", key="user_input")

# Process the input when the user submits
if user_input:
    session_id = "unique_session_id"  # Replace with actual session handling logic
    response = conversational_rag_chain.invoke({"input": user_input}, config={"configurable": {"session_id": session_id}})
    
    # Display streaming response properly
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # Streaming logic
        for chunk in response["answer"]:  
            full_response += chunk  
            message_placeholder.markdown(full_response)

    # Store conversation history
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Bot", full_response))  # Use full_response instead of response["answer"]

# Display chat history
st.subheader("Chat History")
for sender, message in st.session_state.chat_history:
    st.markdown(f"**{sender}:** {message}")

