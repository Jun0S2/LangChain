This project demonstrates how to process text data using LangChain and convert it into vector embeddings with OpenAIEmbeddings. These embeddings are then stored in Pinecone, a popular vector store, to enable efficient similarity search and other operations. The overall workflow is:
Split: Divide the text into smaller chunks.
Ingest: Process the chunks to prepare them for embedding.
Store in Vector: Save the generated vector embeddings in Pinecone.

# Environement Setup

```
pip install langchain langchain-openai langchain-pinecone python-dotenv langchainghub black
```

- langchain: A framework to interact with large language models (LLMs) and manage the data processing pipeline.
- langchain-openai: Provides integration with OpenAI models.
- langchain-pinecone: Simplifies the connection between LangChain and Pinecone.
- python-dotenv: Loads environment variables from a `.env` file.
- langchainghub: Contains additional functionalities and tools related to LangChain.
- black: A code formatter that ensures consistent code style across the project

# Pinecone

Pinecone is a cloud-based `vector database` designed for storing and querying high-dimensional vectors efficiently. Key details include:

- website : [https://www.pinecone.io](https://www.pinecone.io)
- Create new Index / medium-blogs-embedding-index
- Dimensions : 1,536
- Metric : cosine (default)
- Cloud provider : aws
  Now, add it in your .env file as well
- (medium-blog)[https://medium.com/@EjiroOnose/vector-database-what-is-it-and-why-you-should-know-it-ae7e7dca82a4#:~:text=Vector%20databases%20can%20be%20used,not%20all%20vectors%20are%20embeddings.]

Before running the project, add your Pinecone API key and index name to the .env file:

```
# .env
INDXE_NAME = "medium-blogs-embedding-index"
PINECONE_API_KEY
```

# Data Processing Workflow

> split -> ingest-> store in vector

1. Text Splitting

   Large blocks of text are split into smaller, manageable chunks. This helps the language model process the data efficiently without exceeding token limits.

2. Data Ingestion

   The text chunks are loaded and prepared for further processing (e.g., embedding generation). This stage ensures that the text is in a format that the LLM can understand.

3. Embedding and Storage
   Using OpenAIEmbeddings, the processed text chunks are converted into high-dimensional vectors. These embeddings are then stored in the Pinecone vector store for future retrieval and similarity searches.

# Key Components

## TextLoader

The TextLoader is used to load text data from various sources (e.g., files, URLs) and convert it into a format that is easily processed by the language model.

(Text Loader Document)[https://python.langchain.com/v0.1/docs/modules/data_connection/document_loaders/]

## Text Splitter

This component splits the loaded text into smaller segments (chunks) to ensure they are of a manageable size for processing by the language model.

## OpenAIEmbeddings

Converts the text chunks into vector embeddings using OpenAI's models.

### Usage

The resulting embeddings capture the semantic meaning of the text and can be stored in Pinecone for tasks such as similarity search, recommendation, and more.

# RAG(Retrieval-Augmented Generation)

It is a technique that enhances the capabilities of language models by **retrieving relevant information from external sources** (such as documents or databases) and using that data to generate more accurate and contextually enriched responses.

## Retrieval:

A retrieval system (often connected to a vector store like Pinecone) searches and selects relevant documents based on the input query.

## Generation:

The language model then uses both the original query and the retrieved documents to generate a response that is informed by the external data.

### Integration with Document Loaders:

> RAG and Document Loaders?
> Pipeline involves loading external documents (using tools like a TextLoader) and then leveraging those documents in the retrieval step. The model accesses this external information to produce more comprehensive outputs.

## Why Use RAG?

### Up-to-Date Information:

Language models are trained on data up to a certain point in time. RAG allows them to incorporate current data from external sources.

### Context-Aware Generation:

By retrieving context-specific documents, the model can generate responses that are more detailed and better tailored to the query.

### Improved Reliability:

The additional context from retrieved documents helps reduce hallucinations (i.e., generating plausible but incorrect information) by grounding the output in actual external data.
