import os
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
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
client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)
COLLECTION_NAME = "insurance_policies"

insurance_text = """
[ClaimGuard Health Insurance - Policy Document 2026]

Section 1: Outpatient Services
Clause 1.1 - General Practitioner (GP) Visits: Covered up to $150 per visit, maximum 10 visits per year.
Clause 1.2 - Specialist Consultations: Covered up to $300 per visit, requires GP referral.
Clause 1.3 - Acute Infections: For treatments of acute respiratory or throat infections (including prescriptions and consultations), the maximum coverage is $1,500 per incident.

Section 2: Diagnostic Tests
Clause 2.1 - Blood Tests: Fully covered if prescribed by a network doctor.
Clause 2.2 - Medical Imaging: X-rays, Ultrasounds, and MRI scans are covered up to $2,000 per year. MRI requires prior authorization.

Section 3: Inpatient & Surgery
Clause 3.1 - Hospital Room: Standard private room is fully covered for a maximum of 30 days per year.
Clause 3.2 - Surgeries: Major surgeries are covered up to $50,000. Cosmetic or elective surgeries are strictly excluded.

Section 4: Physiotherapy & Rehabilitation
Clause 4.1 - Physical Therapy: Covered up to $100 per session, maximum 20 sessions per year. Must be related to a covered accident or surgery.

Section 5: Exclusions
Clause 5.1 - Routine Full Body Check Ups: Health screenings and routine check-ups are not covered under this reimbursement policy.
Clause 5.2 - Dental Care: Dental scaling, whitening, and cosmetic procedures are excluded.
"""

# Text Splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " "]
)
texts = text_splitter.split_text(insurance_text)
docs = [Document(page_content=t) for t in texts]

def build_cloud_db():
    print("Connecting to Qdrant Cloud...")

    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
        )
        print(f"Created a new cloud space: {COLLECTION_NAME}")
    
    print(f"Ready to convert {len(docs)} policy chunks to vectors and upload...")

    QdrantVectorStore.from_documents(
        docs,
        embeddings_model,
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        collection_name=COLLECTION_NAME
    )

    print("Task completed! Insurance policy knowledge base has been deployed to the Cloud.")

if __name__ == "__main__":
    build_cloud_db()