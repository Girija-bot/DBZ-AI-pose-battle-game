# rag_engine.py

from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain_community.embeddings import SentenceTransformerEmbeddings
from knowledge_base import DBZ_MOVES
import os

CHROMA_DIR = "./chroma_db"

def build_vector_store():
    """Build the ChromaDB vector store from DBZ moves knowledge base."""
    
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    documents = []
    for move in DBZ_MOVES:
        content = f"""
        Move: {move['move']}
        User: {move['user']}
        Description: {move['description']}
        Pose Cues: {move['pose_cues']}
        Scoring Criteria: {move['scoring_criteria']}
        Power Level: {move['power_level']}
        """
        doc = Document(
            page_content=content,
            metadata={"move": move["move"], "power_level": move["power_level"]}
        )
        documents.append(doc)
    
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    print("✅ Vector store built successfully!")
    return vectorstore


def load_vector_store():
    """Load existing vector store or build a new one."""
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    if os.path.exists(CHROMA_DIR):
        vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings
        )
    else:
        vectorstore = build_vector_store()
    
    return vectorstore


def retrieve_move_info(query: str, vectorstore, k=1):
    """Retrieve the most relevant DBZ move for a given pose query."""
    results = vectorstore.similarity_search(query, k=k)
    if results:
        return results[0].page_content
    return "No matching move found."


if __name__ == "__main__":
    build_vector_store()