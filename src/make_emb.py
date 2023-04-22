import os
import argparse

from langchain.document_loaders import OnlinePDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS


def main(args):
    os.environ["OPENAI_API_KEY"] = args.OPEN_API_KEY
    
    loader = OnlinePDFLoader(args.pdf_url)
    pdf_docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    # Create OpenAIEmbeddings and FAISS objects. Vectorize the chunks created above and save.
    documents = text_splitter.split_documents(pdf_docs)
    embeddings = OpenAIEmbeddings()
    faiss_db = FAISS.from_documents(documents, embeddings)
    faiss_db.save_local(args.faiss_save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--OPEN_API_KEY", type=str, default="")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Please specify the chunk_size for CharacterTextSplitter within a number less than or equal to 4096.")
    parser.add_argument("--chunk_overlap", type=int, default=200, help="Please specify the chunk_overlap for CharacterTextSplitter within a number less than or equal to 4096.")
    parser.add_argument("--faiss_save_path", type=str, default="faiss_pdf_index", help="Please specify the name of the created Faiss object.")
    parser.add_argument("--pdf_url", type=str, default="", help="Please specify the path of the PDF file to be read.")
    args = parser.parse_args()

    main(args)
