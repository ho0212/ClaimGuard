import os
import shutil
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from src.agent.agent_workflow import app as langgraph_agent
from fastapi.middleware.cors import CORSMiddleware

api = FastAPI(title="ClaimGuard AI API", description="Insurance Claims Automated Review System")

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

@api.post("/api/evaluate-claim")
async def evaluate_claim(receipt_image: UploadFile = File(...)):
    """
    This function firstly receives files uploaded from frontend, then parses data to LangGraph for analysis.
    Finally, it returns LLM claim report.
    """
    try:
        print(f"Received files from frontend: {receipt_image.filename}")

        # Create temporary folder for uploaded files
        os.makedirs("tmp_uploads", exist_ok=True)
        tmp_file_path = f"tmp_uploads/{receipt_image.filename}"
        with open(tmp_file_path, "wb") as buffer:
            shutil.copyfileobj(receipt_image.file, buffer)
        
        # Add files to AgentState
        initial_state = {"image_path": tmp_file_path}

        print("Agent WorkFlow Starts")
        final_state = langgraph_agent.invoke(initial_state)

        # Clear uploaded files
        os.remove(tmp_file_path)

        return JSONResponse(content={
            "status": "success",
            "extracted_data": final_state.get("extracted_data"),
            "policy_matched": final_state.get("policy_text"),
            "decision_report": final_state.get("final_decision")
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={
            "status": "error",
            "message": str(e)
        })



@api.get("/")
def health_check():
    return {"status": "ClaimGuard API is running"}