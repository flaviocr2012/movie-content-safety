import os
import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Project structure
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "knowledge_base.csv")
INDEX_PATH = os.path.join(PROJECT_ROOT, "data", "faiss_index")

def load_knowledge_base(file_path: str = DATA_PATH) -> list[Document]:
    """
    Load the knowledge base CSV and convert to LangChain Documents.
    """
    # Read the CSV with proper quoting
    df = pd.read_csv(file_path, quoting=1)  # quoting=1 means QUOTE_ALL

    # Convert each row to a Document
    documents = []
    for _, row in df.iterrows():
        content = f"Question: {row['question']}\nAnswer: {row['answer']}"
        metadata = {
            "question": row['question'],
            "answer": row['answer']
        }
        doc = Document(page_content=content, metadata=metadata)
        documents.append(doc)

    print(f"✅ Loaded {len(documents)} knowledge base entries")
    return documents

def create_vector_store(documents: list[Document], save_path: str = INDEX_PATH):
    """
    Generate embeddings and create FAISS vector store.

    Args:
        documents: List of Document objects
        save_path: Directory to save the FAISS index

    Returns:
        FAISS vector store object
    """
    # Initialize the embedding model
    print("🔄 Initializing embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # Create FAISS vector store
    print("🔄 Creating FAISS vector store...")
    vector_store = FAISS.from_documents(documents, embeddings)

    # Save the index locally
    print(f"💾 Saving FAISS index to: {save_path}")
    vector_store.save_local(save_path)

    print("✅ Vector store created and saved successfully!")
    return vector_store

def load_vector_store(load_path: str = INDEX_PATH):
    """
    Load a previously saved FAISS vector store.

    Args:
        load_path: Directory where the FAISS index is saved

    Returns:
        FAISS vector store object
    """
    print(f"🔄 Loading FAISS index from: {load_path}")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    vector_store = FAISS.load_local(
        load_path,
        embeddings,
        allow_dangerous_deserialization=True
    )
    print("✅ FAISS index loaded successfully!")
    return vector_store

def main():
    """
    Main entry point for building the vector store.
    """
    print("=" * 60)
    print("📚 Building Knowledge Base Vector Store")
    print("=" * 60)

    # Step 1: Load the knowledge base
    print("\n📖 Step 1: Loading knowledge base...")
    documents = load_knowledge_base()

    # Step 2: Create and save vector store
    print("\n🔧 Step 2: Creating vector store...")
    vector_store = create_vector_store(documents)

    # Step 3: Verify it works
    print("\n🧪 Step 3: Testing retrieval...")
    test_query = "Does this movie contain extreme graphic violence?"
    results = vector_store.similarity_search(test_query, k=2)

    print(f"\n📝 Test query: '{test_query}'")
    print(f"✅ Retrieved {len(results)} relevant documents:")
    for i, doc in enumerate(results):
        print(f"\n  Result {i+1}:")
        print(f"  Question: {doc.metadata['question']}")
        print(f"  Answer: {doc.metadata['answer']}")

    print("\n" + "=" * 60)
    print("✅ Vector store build complete!")
    print(f"📁 Index saved to: {INDEX_PATH}")
    print("=" * 60)

if __name__ == "__main__":
    main()