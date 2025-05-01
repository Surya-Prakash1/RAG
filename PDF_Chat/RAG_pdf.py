
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from dotenv import load_dotenv


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    # Using smaller chunks for more precise retrieval
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Smaller chunk size
        chunk_overlap=200,  # Reasonable overlap to maintain context
        separators=["\n\n", "\n", ".", " ", ""],  # More nuanced splitting
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store


def get_enhanced_retriever(embeddings):
    # Load the vector store
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    # Create a basic retriever
    basic_retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 6}  # Retrieve more documents for better context
    )
    
    # Create an LLM for contextual compression
    llm = Ollama(model="llama3.2", temperature=0)
    
    # Create a document compressor that uses an LLM to extract only the relevant parts of documents
    compressor = LLMChainExtractor.from_llm(llm)
    
    # Create a compressed retriever that first retrieves documents then extracts relevant parts
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=basic_retriever
    )
    
    return compression_retriever


def get_conversational_chain():
    # More detailed prompt template with better instructions
    prompt_template = """
    You are an expert AI assistant tasked with providing precise, detailed answers based solely on the provided documents.

    INSTRUCTIONS:
    1. Base your answer EXCLUSIVELY on the context below.
    2. Do NOT introduce any outside information or assumptions.
    3. If the exact answer cannot be found in the context, explicitly state: "I don't have enough information to answer this question based on the provided documents."
    4. Quote specific text from the documents when appropriate, using "quotes" to indicate direct citations.
    5. Include all relevant details, figures, numbers, and specific information found in the context.
    6. Organize your answer logically with clear structure.
    7. Provide complete explanations without summarizing or oversimplifying.

    Context:
    {context}

    Question:
    {question}

    Detailed Answer:
    """

    model = Ollama(model="llama3.2", temperature=0.1)  # Lower temperature for more factual responses

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    return model, prompt


def user_input(user_question):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    # Get the advanced retriever
    retriever = get_enhanced_retriever(embeddings)
    
    # Get relevant documents
    docs = retriever.get_relevant_documents(user_question)
    
    # Display the retrieved chunks for debugging
    with st.expander("Retrieved Content"):
        for i, doc in enumerate(docs):
            st.markdown(f"**Chunk {i+1}:**")
            st.markdown(doc.page_content)
            st.markdown("---")
    
    # Get the model and prompt
    model, prompt = get_conversational_chain()
    
    # Format the context
    context_text = "\n\n".join([doc.page_content for doc in docs])
    
    # Format the prompt with the context and question
    formatted_prompt = prompt.format(context=context_text, question=user_question)
    
    # Generate the response
    response = model.invoke(formatted_prompt)
    
    st.markdown("### Reply:")
    st.markdown(response)


def main():
    st.set_page_config(page_title="Enhanced PDF Chat", layout="wide")
    st.header("Chat with PDF using Llama 3.2")
    st.markdown("Upload PDFs and ask questions to get answers based on their content.")

    # Initialize session state for tracking whether documents have been processed
    if 'docs_processed' not in st.session_state:
        st.session_state.docs_processed = False

    # Create two columns for layout
    col1, col2 = st.columns([2, 1])

    with col2:
        st.markdown("### Document Upload")
        pdf_docs = st.file_uploader("Upload PDF Files", accept_multiple_files=True)
        
        if st.button("Process Documents"):
            if pdf_docs:
                with st.spinner("Processing documents..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.session_state.docs_processed = True
                    st.success("Documents processed successfully!")
            else:
                st.error("Please upload at least one PDF document.")
    
    with col1:
        st.markdown("### Ask Questions")
        user_question = st.text_input("Enter your question about the documents")
        
        if user_question:
            if st.session_state.docs_processed:
                with st.spinner("Searching for information..."):
                    user_input(user_question)
            else:
                st.warning("Please upload and process documents before asking questions.")


if __name__ == "__main__":
    main()

