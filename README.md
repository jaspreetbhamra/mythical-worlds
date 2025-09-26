# Mythical Worlds

This project is a Retrieval-Augmented Generation (RAG) pipeline designed to answer questions about a collection of mythological and fantasy texts from Project Gutenberg.

## Things I Learnt
1. What does a RAG pipeline look like?
2. Vector databases
    - FAISS
    - Chroma
    - Pros and Cons
    - Search methods - L2, Cosine, Dot
    - Indexing methods - Exact, ANN
3. Text foundational models optimized for embeddings
4. Answer generation based on embedddings using full LLMs
5. Basic front end using Streamlit (full list of technoloiges below)

## Workflow

The project follows these steps:

1.  **Data Ingestion**: Downloads public domain texts from Project Gutenberg.
2.  **Preprocessing**: Cleans the raw text files by removing headers and footers, then splits them into smaller, manageable chunks.
3.  **Embedding**: Uses a sentence transformer model to generate vector embeddings for each text chunk.
4.  **Indexing**: Stores the embeddings in a FAISS index for efficient similarity search.
5.  **Retrieval**: When a user asks a question, the system embeds the query and retrieves the most relevant text chunks from the FAISS index.
6.  **Generation**: The retrieved chunks are then used as context for a large language model (LLM) which generates an answer to the user's question.

## Technologies and Packages Used

*   **Python**: The core programming language.
*   **Streamlit**: For the web-based user interface.
*   **Ollama**: To run the local LLM for text generation.
*   **SentenceTransformers**: For creating text embeddings.
*   **FAISS**: For efficient similarity search of vector embeddings.
*   **Project Gutenberg**: As the source of the text data.
*   **Conda**: For environment management.

## How to Reproduce

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd mythical-worlds
    ```

2.  **Create the Conda environment:**
    ```bash
    conda env create -f environment.yml
    conda activate mythical-worlds
    ```

3.  **Download the data:**
    ```bash
    python src/ingestion/download_gutenberg.py
    ```

4.  **Preprocess the data:**
    ```bash
    python src/ingestion/preprocess.py
    ```

5.  **Embed the text chunks and build the FAISS index:**
    ```bash
    python src/embeddings/embed_chunks.py
    ```

## How to Run

You can interact with the Mythical Worlds Explorer in two ways:

### 1. Streamlit Web App

This provides a user-friendly interface to ask questions.

```bash
streamlit run src/ui/app.py
```

### 2. Command-Line Interface

For a more direct way to query the system.

```bash
python src/retrieval/rag_pipeline.py "Your question here" --model <model_name>
```

For example:
```bash
python src/retrieval/rag_pipeline.py "Who was Odin in Norse mythology?" --model mistral
```