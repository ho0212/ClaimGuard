import os
import shutil
from typing import List, Annotated
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from src.agent.agent_workflow import app as langgraph_agent
from fastapi.middleware.cors import CORSMiddleware
import fitz
import uuid

api = FastAPI(title="ClaimGuard AI API", description="Insurance Claims Automated Review System")

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

@api.post("/api/evaluate-claim")
async def evaluate_claim(
    receipt_files: Annotated[List[UploadFile], File(description="Upload receipt files for claim evaluation")],
    company: Annotated[str, Form(description="Insurance Company")],
    plan: Annotated[str, Form(description="Insurance Plan")]
):
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
            file_extension = file.filename.split('.')[-1].lower()
            file_content = await file.read()

            # If the file is a PDF, extract each page as an image for VLM processing
            if file_extension == "pdf":
                pdf_doc = fitz.open(stream=file_content, filetype="pdf")
                for page_num in range(pdf_doc.page_count):
                    page = pdf_doc.load_page(page_num)
                    image = page.get_pixmap(dpi=150)

                    tmp_image_path = f"temp_{uuid.uuid4()}_page_{page_num}.png"
                    image.save(tmp_image_path)
                    tmp_file_paths.append(tmp_image_path)
            else:
                tmp_file_path = f"temp_{uuid.uuid4()}.{file_extension}"
                with open(tmp_file_path, "wb") as buffer:
                    buffer.write(file_content)
                tmp_file_paths.append(tmp_file_path)

        
        initial_state = {"files": tmp_file_paths, "company": company, "plan": plan}

        print("Passing files to LangGraph agent for analysis...")
        final_state = langgraph_agent.invoke(initial_state)

        # Clean up temporary files
        for path in tmp_file_paths:
            os.remove(path)
        
        # # === Test Section ===
        # print("LLM Decision Report:", final_state.get("final_decision"))

        # # === End of Test Section ===

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