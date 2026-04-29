import os
import base64
from typing import List, TypedDict
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from openai import AzureOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langgraph.graph import StateGraph, START, END
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

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
COLLECTION_NAME = "health_insurance_policies"

# Define JSON format from VLM
class DocumentData(BaseModel):
    is_readable: bool = Field(description="If the image is too blurry to read, set to False.")
    doc_type: str = Field(description="Must be either 'receipt' or 'diagnosis_doc'")
    date: str = Field(default="Unknown")
    diagnosis: str = Field(default="Unknown")
    cost: float = Field(default=0.0)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    
# === Define State ===
class AgentState(TypedDict):
    files: List[str]            # List of file paths for uploaded images
    extracted_items: List[dict] # JSON data extracted from each files
    final_diagnosis: str        # Final diagnosis extracted from the receipt
    aggregated_cost: float      # Aggregated cost from all receipts
    validation_status: str      # Validation status: PASSED or FAILED
    policy_text: str
    final_decision: str

# === Define Nodes ===
def node_extract_vision(state: AgentState):
    print("Node 1: VLM")
    print("Scanning receipt and extracting relevant information...")
    extracted_list = []
    
    for file_path in state["files"]:
        print(f"Processing file: {os.path.basename(file_path)}")
        base64_image = encode_image(file_path)

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
            response_format=DocumentData
        )
        parsed_data = response.choices[0].message.parsed.model_dump()
        extracted_list.append(parsed_data)

    return {"extracted_items": extracted_list}

def node_cross_validation(state: AgentState):
    print("Node 2: Cross Validation")
    total_cost = 0
    main_diagnosis = "Unknown"

    for item in state["extracted_items"]:
        if not item.get("is_readable"):
            print("Error: Detected unreadable image. Failing the claim.")
            return {"validation_status": "FAILED", "final_decision": "Claim Denied due to unreadable files. Please provide clearer images."}
        
        if item.get("doc_type") == "receipt":
            total_cost += item.get("cost", 0.0)
            print(f"Added ${item.get('cost', 0.0)} to total cost. Current total: ${total_cost}")
        elif item.get("doc_type") == "diagnosis_doc":
            main_diagnosis = item.get("diagnosis", "Unknown")
            print(f"Extracted diagnosis: {main_diagnosis}")
    
    if total_cost == 0 or main_diagnosis == "Unknown" or main_diagnosis == "N/A":
        print("Error: Missing critical information. Failing the claim.")
        return {"validation_status": "FAILED", "final_decision": "Claim Denied due to missing critical information. Please ensure all files are clear and complete."}

    print("Cross-validation passed.")
    return {
        "validation_status": "PASSED",
        "final_diagnosis": main_diagnosis,
        "aggregated_cost": total_cost
    }


def node_retrieve_policy(state: AgentState):
    print("Node 3: RAG")
    print("Retrieving relevant policy clauses...")
    diagnosis = state["final_diagnosis"]

    # === Hardcoded parameters for now ===
    target_company = "Allianz"
    target_plan = "OVHC"
    print(f"Filtering for company: {target_company} and plan: {target_plan}")
    # === ===

    qdrant_filter = Filter(
        must=[
            FieldCondition(
                key="metadata.company",
                match=MatchValue(value=target_company)
            ),
            FieldCondition(
                key="metadata.plan",
                match=MatchValue(value=target_plan)
            )
        ]
    )

    # Initialise vector database
    vector_store = QdrantVectorStore(
        client=qdrant_client,
        embedding=embeddings_model,
        collection_name=COLLECTION_NAME
    )

    results = vector_store.similarity_search_with_score(diagnosis, k=3, filter=qdrant_filter)

    # filter irrelevant info
    SCORE_THRESHOLD = 0.2
    filtered_results = [(doc, score) for doc, score in results if score >= SCORE_THRESHOLD]

    if filtered_results:
        matched_text = "\n".join([f"- {doc.page_content}" for doc, _ in filtered_results])
        matched_score = [score for _, score in filtered_results]
    else:
        matched_text = "No relevant policy found."
        matched_score = []
    
    # print(f"Relevant Policy Found: \n{matched_text}")
    print(f"Corresponding Score: {matched_score}")

    return {"policy_text": matched_text}



def node_make_decision(state: AgentState):
    print("Node 4: LLM Decision")
    print("Evaluating results and generating claim report...")

    # Integrate results from previous steps
    prompt = f"""
    You are an expert Health Insurance Claims Adjuster.
    
    [Aggregated Claim Data]:
    - Primary Diagnosis: {state['final_diagnosis']}
    - Total Billed Cost: ${state['aggregated_cost']}
    
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

def routing_logic(state: AgentState):
    if state["validation_status"] == "FAILED":
        print("Data validation failed. Routing to decision node for claim denial.")
        return END
    else:
        return "Policy_Retriver"

# === Build the Graph ===
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("Vision_Extractor", node_extract_vision)
workflow.add_node("Cross_Validator", node_cross_validation)
workflow.add_node("Policy_Retriver", node_retrieve_policy)
workflow.add_node("Decision_Maker", node_make_decision)

# Add edges
workflow.add_edge(START, "Vision_Extractor")
workflow.add_edge("Vision_Extractor", "Cross_Validator")
workflow.add_conditional_edges("Cross_Validator", routing_logic)
workflow.add_edge("Policy_Retriver", "Decision_Maker")
workflow.add_edge("Decision_Maker", END)

app = workflow.compile()