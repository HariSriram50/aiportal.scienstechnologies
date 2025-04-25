from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from openai import OpenAI
import json

# ✅ Your API key directly used here (replace with your key)
client = OpenAI(api_key="")

router = APIRouter()

class QuestionRequest(BaseModel):
    num_questions: int 
    job_description: str
    difficulty_level: str
    type_questions: str

@router.post("/generate-questions")
async def generate_questions(data: QuestionRequest):
    try:
        prompt = f"""
        Generate {data.num_questions} {data.type_questions.lower()} interview questions 
based on the following job description.

These questions should match the difficulty level: {data.difficulty_level.lower()}.

Return the response in JSON format as shown:
{{
    "difficulty_level": "{data.difficulty_level}",
    "type": "{data.type_questions}",
    "questions": [
        "Question 1?",
        "Question 2?",
        ...
    ]
}}
        Job Description:
        {data.job_description}
        """

        response = client.chat.completions.create(
            model="gpt-4",  # 🧠 GPT-4 directly
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=600
        )
        reply = json.loads(response.choices[0].message.content.strip())["questions"]
        finalresponse = {
            "difficulty_level": data.difficulty_level,
            "type": data.type_questions,
            "questions": reply
        }

    
        return {"success": True, "data": finalresponse}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ Error: {str(e)}")
