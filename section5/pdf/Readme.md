# PDF Document Search with FAISS and LangChain

This project demonstrates how to use FAISS (a local vector search engine) and LangChain to load, process, and query a PDF document. The document is embedded using OpenAI's embedding model, stored in FAISS, and queried with an LLM to generate natural language answers.

## Features

### PDF Loading

Load and process PDF documents using LangChain's PyPDFLoader.

### Text Splitting

Split the document into manageable chunks for better embedding and retrieval.

### Vector Storage

Use FAISS to store and retrieve document embeddings locally.

### Question Answering

Query the document using natural language with LangChain's retrieval and document combination chains.

## Technologies Used

# LangChain

Framework for building applications with LLMs.

# FAISS

Facebook AI Similarity Search for efficient similarity search and clustering of dense vectors.

# OpenAI

For generating embeddings and language model responses.

## Setup Instructions

### Clone the Repository

```
git clone https://github.com/your-repo/langchain-faiss-pdf
cd langchain-faiss-pdf
```

### Install Dependencies

Ensure you have Python 3.9+ installed. Then install the required libraries:

```
pip install langchain openai faiss-cpu python-dotenv

```

### Set Up Environment Variables

Create a `.env` file in the project root and add your OpenAI API key:

```
OPENAI_API_KEY=your_openai_api_key
```

### Add Your PDF Document

Place your PDF file (e.g., react.pdf) in a directory, and update the file path in the script accordingly:
pdf_path = "/path/to/your/react.pdf"

## How It Works

### Load PDF Document

The script uses PyPDFLoader to extract text from the PDF.

### Split Text into Chunks

The document is split into chunks of 1000 characters with an overlap of 30 characters for better retrieval performance.

### Embed and Store in FAISS

Each chunk is converted into a vector using OpenAI Embeddings and stored in a FAISS index.

### Query and Retrieve Information

The FAISS index is queried using LangChain's retrieval chain, and the retrieved documents are processed with an LLM to generate natural language answers.

## Running the Script

```
python main.py
```

You should see the summarized response of your query, e.g.,

```
ReAct is a framework that combines reasoning and acting by allowing language models to both generate thoughts and take actions. It improves performance in complex tasks by enabling iterative reasoning steps. ReAct is effective for tasks like question answering, reasoning, and real-world applications.
```

### Customization

- Change the PDF File: Update the `pdf_path` variable with your own PDF file path.
- Modify the Query: Change the `input` parameter in the `invoke()` method to ask different questions.
