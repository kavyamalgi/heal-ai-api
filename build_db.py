import os
import zipfile
import logging
from pathlib import Path

from pypdf import PdfReader
from dotenv import load_dotenv

# LangChain + FAISS imports
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document

# --- Configuration ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

ZIP_FILE_NAME = "Gold_Standard.zip"
EXTRACT_TO_DIRECTORY = "gold_standard_docs"
VECTOR_DB_PATH = "faiss_diseases_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def unzip_local_file(zip_filename, extract_dir):
    """Unzips a local file if the destination directory doesn't exist."""
    if not os.path.exists(zip_filename):
        logging.error(f"'{zip_filename}' not found in the project directory.")
        return False

    if os.path.exists(extract_dir):
        logging.info(f"Directory '{extract_dir}' already exists. Skipping extraction.")
        return True

    logging.info(f"Unzipping '{zip_filename}' to '{extract_dir}'...")
    os.makedirs(extract_dir, exist_ok=True)

    with zipfile.ZipFile(zip_filename, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

    logging.info("Unzip complete.")
    return True


def extract_text_from_pdf(pdf_path):
    """Extracts text from a single PDF file."""
    try:
        reader = PdfReader(pdf_path)
        text = ""

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

        return text.strip()

    except Exception as e:
        logging.warning(f"Could not read PDF '{pdf_path}': {e}")
        return ""


def load_documents_from_directory(directory):
    """Loads and converts all PDFs in a directory into LangChain Document objects."""
    documents = []

    pdf_files = list(Path(directory).rglob("*.pdf"))
    logging.info(f"Found {len(pdf_files)} PDF files in '{directory}'.")

    for pdf_path in pdf_files:
        text = extract_text_from_pdf(pdf_path)

        if text:
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": str(pdf_path),
                        "title": pdf_path.stem
                    }
                )
            )

    logging.info(f"Successfully loaded {len(documents)} documents.")
    return documents


def main():
    """Main function to build the vector database from a local zip file."""
    if not unzip_local_file(ZIP_FILE_NAME, EXTRACT_TO_DIRECTORY):
        return

    # 1. Load documents from the extracted directory
    documents = load_documents_from_directory(EXTRACT_TO_DIRECTORY)

    if not documents:
        logging.error("No documents were loaded. Aborting.")
        return

    # 2. Split documents into chunks
    logging.info("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    logging.info(f"Created {len(chunks)} text chunks.")

    # 3. Initialize local embedding model
    logging.info("Initializing embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    # 4. Create and save the FAISS vector database
    logging.info("Building FAISS vector database...")
    vector_db = FAISS.from_documents(chunks, embeddings)

    vector_db.save_local(VECTOR_DB_PATH)
    logging.info(f"Vector database built and saved successfully at '{VECTOR_DB_PATH}'.")


if __name__ == "__main__":
    main()
