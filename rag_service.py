import os
import logging
from typing import List, Dict, Any, Optional

from langchain_community.vectorstores import FAISS
from langchain_anthropic import ChatAnthropic
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- Constants ---
VECTOR_DB_PATH = "faiss_diseases_db"

# Local embedding model (no API key)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Claude model
LLM_MODEL_NAME = "claude-3-5-sonnet-20241022"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class RAGService:
    """Encapsulates the core Retrieval-Augmented Generation logic."""

    def __init__(self):
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found. Please ensure it's in your .env file.")

        logging.info("Initializing RAG Service...")

        self.embeddings = self._initialize_embeddings()
        self.vector_db = self._load_vector_db()
        self.llm = self._initialize_llm()
        self.rag_chain = self._create_rag_chain()

        logging.info("RAG Service Initialized Successfully.")

    def _initialize_embeddings(self) -> HuggingFaceEmbeddings:
        return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    def _initialize_llm(self) -> ChatAnthropic:
        # ChatAnthropic will read ANTHROPIC_API_KEY from env, but we pass explicitly for clarity
        return ChatAnthropic(
            model=LLM_MODEL_NAME,
            temperature=0.2,
            anthropic_api_key=self.api_key,
        )

    def _load_vector_db(self) -> FAISS:
        if not os.path.exists(VECTOR_DB_PATH):
            raise FileNotFoundError(
                f"Vector DB not found at '{VECTOR_DB_PATH}'. Run build_db.py first."
            )

        logging.info(f"Loading vector DB from {VECTOR_DB_PATH}...")
        return FAISS.load_local(
            VECTOR_DB_PATH,
            self.embeddings,
            allow_dangerous_deserialization=True
        )

    def _create_rag_chain(self) -> Optional[RetrievalQA]:
        retriever = self.vector_db.as_retriever(search_kwargs={"k": 5})
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )

    def process_query(self, patient_history: str, conversation_history: List[str], query: str) -> Dict[str, Any]:
        if not self.rag_chain:
            return {"error": "RAG chain not initialized."}

        formatted_history = "\n".join(conversation_history)
        structured_query = (
            f"**Static Patient History:**\n{patient_history}\n\n"
            f"**Ongoing Conversation History:**\n{formatted_history}\n\n"
            f"**Clinician's Latest Request:**\n{query}\n\n"
            f"**Task:**\nBased on ALL the information, provide a concise and clinically relevant answer."
        )

        try:
            logging.info("Invoking RAG chain...")
            raw_outputs = self.rag_chain.invoke({"query": structured_query})

            sources = []
            for doc in raw_outputs.get("source_documents", []):
                meta = doc.metadata
                sources.append({
                    "source": meta.get("source"),
                    "url": meta.get("url"),
                    "title": meta.get("title", meta.get("doc_id"))
                })

            return {
                "answer": raw_outputs.get("result", ""),
                "sources": sources
            }

        except Exception as e:
            logging.error(f"RAG chain invocation failed: {e}")
            return {"error": f"Failed to process query: {e}"}
