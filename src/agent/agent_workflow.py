import os
import base64
from typing import TypedDict
from dotenv import load_dotenv
from pydantic import BaseModel
from openai import AzureOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langgraph.graph import StateGraph, START, END

load_dotenv()


# Initialise Azure and Qdrant clients (VLM, LLM, embedding model, Qdrant)
client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)

embeddings_model = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
)

qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)
COLLECTION_NAME = "insurance_policies"

# Define JSON format from VLM
class ReceiptData(BaseModel):
    date: str
    diagnosis: str
    cost: float

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    
# === Define State ===
class AgentState(TypedDict):
    image_path: str
    extracted_data: dict # output from VLM
    policy_text: str # output from RAG
    final_decision: str

# === Define Nodes ===
def node_extract_vision(state: AgentState):
    print("Node 1: VLM")
    print("Scanning receipt and extracting relevant information...")
    base64_image = encode_image(state["image_path"])

    response = client.beta.chat.completions.parse(
        model="gpt-4.1-mini-1",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a medical data extraction expert. Extract the date, cost, and diagnosis from the uploaded receipt. "
                    "For the total cost, always look for the 'TOTAL' including tax and other fees, and IGNORE any 'Amount DUE' or 'Sub Total Figures. '"
                    "Crucially, regardless of whether the original receipt is in English, Traditional Chinese, or any other language, "
                    "you MUST translate and output the JSON values in English. (e.g., if you see '物理治療', output 'Physical Therapy')."
                )
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Please extract the information from this medical receipt."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "high"
                        }
                    }
                ]
            }
        ],
        response_format=ReceiptData
    )

    extracted_json = response.choices[0].message.parsed.model_dump()
    print("Extracted Successfully!")
    print(f"Diagnosis: {extracted_json['diagnosis']}")
    print(f"Cost: {extracted_json['cost']}")
    return {"extracted_data": extracted_json}

def node_retrieve_policy(state: AgentState):
    print("Node 2: RAG")
    print("Retrieving relevant policy clauses...")
    diagnosis = state["extracted_data"]["diagnosis"]

    # Initialise vector database
    vector_store = QdrantVectorStore(
        client=qdrant_client,
        embedding=embeddings_model,
        collection_name="insurance_policies"
    )

    results = vector_store.similarity_search_with_score(diagnosis, k=3)

    # filter irrelevant info
    SCORE_THRESHOLD = 0.3
    filtered_results = [(doc, score) for doc, score in results if score >= SCORE_THRESHOLD]

    if filtered_results:
        matched_text = "\n".join([f"- {doc.page_content}" for doc, _ in filtered_results])
        matched_score = [score for _, score in filtered_results]
    else:
        matched_text = "no relevant policy found."
    
    print(f"Relevant Policy Found: \n{matched_text}")
    print(f"Corresponding Score: {matched_score}")

    return {"policy_text": matched_text}



def node_make_decision(state: AgentState):
    print("Node 3: LLM Decision")
    print("Evaluating results and generating claim report...")

    # Integrate results from previous steps
    prompt = f"""
    You are an expert Health Insurance Claims Adjuster.
    
    [Receipt Data]:
    - Date: {state['extracted_data']['date']}
    - Diagnosis: {state['extracted_data']['diagnosis']}
    - Billed Cost: ${state['extracted_data']['cost']}
    
    [Relevant Policy Clause]:
    {state['policy_text']}
    
    Analyse the claim based strictly on the policy clause. 
    Provide a final decision stating whether the claim is approved, partially approved, or denied, and explain the exact approved amount.
    """

    response = client.chat.completions.create(
        model="gpt-4.1-mini-1",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.1
    )

    decision = response.choices[0].message.content
    return {"final_decision": decision}

# === Build the Graph ===
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("Vision_Extractor", node_extract_vision)
workflow.add_node("Policy_Retriver", node_retrieve_policy)
workflow.add_node("Decision_Maker", node_make_decision)

# Add edges
workflow.add_edge(START, "Vision_Extractor")
workflow.add_edge("Vision_Extractor", "Policy_Retriver")
workflow.add_edge("Policy_Retriver", "Decision_Maker")
workflow.add_edge("Decision_Maker", END)

app = workflow.compile()