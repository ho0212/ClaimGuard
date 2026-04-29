import os
import shutil
from typing import List, Annotated
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
async def evaluate_claim(receipt_files: Annotated[List[UploadFile], File(description="Upload receipt files for claim evaluation")]):
    """
    This function firstly receives files uploaded from frontend, then parses data to LangGraph for analysis.
    Finally, it returns LLM claim report.
    """
    try:
        print(f"Received {len(receipt_files)} files from frontend.")

        # Create temporary folder for uploaded files
        os.makedirs("tmp_uploads", exist_ok=True)
        tmp_file_paths = []

        for file in receipt_files:
            tmp_file_path = os.path.join("tmp_uploads", file.filename)
            with open(tmp_file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            tmp_file_paths.append(tmp_file_path)
        
        initial_state = {"files": tmp_file_paths}

        print("Passing files to LangGraph agent for analysis...")
        final_state = langgraph_agent.invoke(initial_state)

        # Clean up temporary files
        for path in tmp_file_paths:
            os.remove(path)
        

        return JSONResponse(content={
            "status": "success",
            "extracted_items": final_state.get("extracted_items"),
            "aggregated_cost": final_state.get("aggregated_cost"),
            "final_diagnosis": final_state.get("final_diagnosis"),
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