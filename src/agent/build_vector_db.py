import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

load_dotenv()

# Initialise Azure Embedding model
embeddings_model = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
)

# Initialise Qdrant client
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)
COLLECTION_NAME = "health_insurance_policies"


def setup_collection():
    # Check if collection exists, if not create it
    if not qdrant_client.collection_exists(COLLECTION_NAME):
        print(f"Creating New Qdrant Collection: {COLLECTION_NAME}")
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
        )
    else:
        print(f"Collection '{COLLECTION_NAME}' already exists.")
    
    try:
        qdrant_client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="metadata.company",
            field_schema="keyword"
        )
        qdrant_client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="metadata.plan",
            field_schema="keyword"
        )
        print("Created payload indexes.")
    except Exception as e:
        pass

def get_dynamic_splitter(target_chunk_size=500, overlap_ratio=0.15):
    """
    This function returns a RecursiveCharacterTextSplitter with dynamic chunk size 
    and overlap based on the target chunk size and overlap ratio.
    """
    dynamic_overlap = int(target_chunk_size * overlap_ratio)
    print(f"Using dynamic chunk size: {target_chunk_size} and overlap: {dynamic_overlap}")

    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=target_chunk_size,
        chunk_overlap=dynamic_overlap
    )

def process_and_upload_policy(pdf_path: str, company: str, plan: str):
    """
    This function processes a PDF insurance policy, extracts text, generates embeddings, and uploads to Qdrant.
    """

    print(f"Processing PDF: {pdf_path}")

    # Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"Extracted {len(documents)} pages from the PDF.")

    # Split text into chunks
    text_splitter = get_dynamic_splitter(target_chunk_size=500, overlap_ratio=0.15)
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")

    # # === Debug: Print the content of the first few chunks to verify splitting ===
    # for i, chunk in enumerate(chunks):
    #     if "appendix" in chunk.page_content.lower():
    #         print(f"Found 'appendix' in chunk {i}.")
    #         print("=" * 50)
    #         print(chunk.page_content)
    #         print("=" * 50)
    #         break
    # # === End Debug ===

    # Add metadata tag
    for chunk in chunks:
        chunk.metadata["company"] = company
        chunk.metadata["plan"] = plan
    
    print("Generating embeddings and uploading to Qdrant... (it might take a while for large documents)")
    QdrantVectorStore.from_documents(
        chunks,
        embeddings_model,
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        collection_name=COLLECTION_NAME,
    )
    print("Upload complete.")

if __name__ == "__main__":
    setup_collection()
    
    target_pdf_path = "policies/Allianz/Allianz_OWHC.pdf"
    company = "Allianz"
    plan = "OVHC_work"


    if os.path.exists(target_pdf_path):
        process_and_upload_policy(target_pdf_path, company=company, plan=plan)
    else:
        print(f"PDF file not found at path: {target_pdf_path}. Please check the file path and try again.")
