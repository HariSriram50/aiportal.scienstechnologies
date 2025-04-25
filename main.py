from fastapi import FastAPI,Request
from post_questions import router as post_router
from resume_analysis import router as resume_router
from video_analysis import router as aihr_router
from resume_analysis(chatgpt) import router as openai_router
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn


app = FastAPI()

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"message": f"Unexpected error occurred: {str(exc)}"},
    )

# Enable CORS if needed


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(post_router)
app.include_router(resume_router)
app.include_router(aihr_router)
app.include_router(openai_router)

