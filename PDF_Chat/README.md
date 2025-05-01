# PDF Chat with RAG and Llama 3.2

An advanced PDF document question-answering system built with Streamlit, leveraging RAG (Retrieval Augmented Generation) and Llama 3.2 for accurate, context-aware responses.

## Features

- 📄 Multi-PDF document upload support
- 🔍 Advanced text chunking with recursive character splitting
- 🧠 Enhanced retrieval using FAISS vector store
- 💡 Contextual compression for more relevant responses
- 🎯 Zero-shot question answering
- 🖥️ User-friendly Streamlit interface

## Prerequisites

- Python 3.8+
- Ollama with Llama 3.2 model installed
- Required Python packages (install via pip):
  ```
  streamlit
  PyPDF2
  langchain
  langchain-community
  faiss-cpu
  python-dotenv
  ```

## Installation

1. Clone this repository
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure Ollama is installed and the Llama 3.2 model is available



## Usage

1. Run the Streamlit application:
   ```bash
   streamlit run RAG_pdf.py
   ```
2. Upload one or more PDF documents using the file uploader
3. Click "Process Documents" to analyze and index the content
4. Enter your questions in the text input field
5. View detailed responses based on the document content

## How It Works

1. **Document Processing**:
   - Extracts text from uploaded PDFs
   - Splits text into manageable chunks with context overlap
   - Creates embeddings using Nomic's text embedding model
   - Stores vectors in a FAISS index for efficient retrieval

2. **Question Answering**:
   - Uses contextual compression for better document retrieval
   - Implements similarity search with enhanced context
   - Generates responses using Llama 3.2 with zero temperature for consistency
   - Displays retrieved context chunks for transparency

## Architecture

- `get_pdf_text()`: Extracts text from PDF documents
- `get_text_chunks()`: Splits text into optimized chunks
- `get_vector_store()`: Creates and manages FAISS vector store
- `get_enhanced_retriever()`: Implements contextual compression retrieval
- `get_conversational_chain()`: Sets up the QA chain with custom prompting
- `user_input()`: Handles user queries and generates responses

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

This project is open source and available under the MIT License.
