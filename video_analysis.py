from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import os
import time
import json
import wave
import cv2
import numpy as np
from io import BytesIO
from moviepy import VideoFileClip
import soundfile as sf
from scipy.signal import resample
from vosk import Model, KaldiRecognizer
import openai
import pdfkit
from pathlib import Path

# Define the router
router = APIRouter()
OPENAI_API_KEY=""

# OpenAI configuration
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Directory setup
UPLOAD_DIR = Path("uploads")
TEMP_DIR = Path("temp")
RESULTS_DIR = Path("results")

for dir_path in [UPLOAD_DIR, TEMP_DIR, RESULTS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Pydantic models
class AnalysisParams(BaseModel):
    worker_role: str = "job candidate"
    evaluation_metrics: str = "Communication Skills, Technical Knowledge, Problem-Solving, Safety Awareness, Teamwork"
    summary_length: str = "medium"
    report_format: str = "tabular with scores from 1-5 for each metric"

class AnalysisResult(BaseModel):
    transcript: str
    dominant_emotion: Optional[str] = None
    body_movement: Optional[str] = None
    emotion_analysis: Dict[str, Any]
    hr_analysis: Dict[str, Any]
    html_report_path: Optional[str] = None
    pdf_report_path: Optional[str] = None

# Routes
@router.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <head>
            <title>HR Video Analysis API</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                h1 { color: #2c3e50; }
                code { background-color: #f5f5f5; padding: 2px 4px; border-radius: 4px; }
                pre { background-color: #f5f5f5; padding: 10px; border-radius: 4px; overflow-x: auto; }
                .endpoint { margin-bottom: 20px; border-left: 4px solid #3498db; padding-left: 10px; }
            </style>
        </head>
        <body>
            <h1>HR Video Analysis API</h1>
            <p>Upload and analyze interview videos to get complete results immediately:</p>
            <div class="endpoint">
                <h3>POST /api/analyze-video</h3>
                <p>Upload a video for analysis and get results immediately:</p>
                <pre>curl -X POST "http://localhost:8000/video/api/analyze-video" -F "video=@interview.mp4"</pre>
            </div>
            <div class="endpoint">
                <h3>GET /api/download-report/{result_id}</h3>
                <p>Download HTML or PDF report:</p>
                <pre>curl -X GET "http://localhost:8000/video/api/download-report/{result_id}?format=pdf" -o report.pdf</pre>
            </div>
        </body>
    </html>
    """

@router.post("/api/analyze-video")
async def analyze_video(
    video: UploadFile = File(...),
    params: Optional[str] = Form("{}")
):
    try:
        analysis_params = AnalysisParams(**json.loads(params))
        video_id = f"{int(time.time())}_{video.filename}"
        video_path = UPLOAD_DIR / video_id
        
        # Save the uploaded video
        with open(video_path, "wb") as f:
            f.write(await video.read())
        
        result_id = f"result_{int(time.time())}"
        
        # Process the video synchronously
        analysis_result = await process_video(str(video_path), result_id, analysis_params)
        
        # Return complete analysis results in JSON format
        return JSONResponse({
            "status": "complete",
            "result_id": result_id,
            "results": analysis_result
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during analysis: {str(e)}")

@router.get("/api/download-report/{result_id}")
async def download_report(result_id: str, format: str = "html"):
    result_file = RESULTS_DIR / f"{result_id}.json"
    if not result_file.exists():
        raise HTTPException(status_code=404, detail="Result not found")
    
    if format.lower() == "html":
        html_path = RESULTS_DIR / f"{result_id}.html"
        if not html_path.exists():
            raise HTTPException(status_code=404, detail="HTML report not found")
        return FileResponse(html_path, media_type="text/html", filename=f"{result_id}_report.html")
    elif format.lower() == "pdf":
        pdf_path = RESULTS_DIR / f"{result_id}.pdf"
        if not pdf_path.exists():
            raise HTTPException(status_code=404, detail="PDF report not found")
        return FileResponse(pdf_path, media_type="application/pdf", filename=f"{result_id}_report.pdf")
    else:
        raise HTTPException(status_code=400, detail="Invalid format. Use 'html' or 'pdf'")

# Helper functions
async def process_video(video_path: str, result_id: str, params: AnalysisParams):
    try:
        audio_path = TEMP_DIR / f"{result_id}_audio.wav"
        transcript_text = ""
        dominant_emotion = "Not analyzed"
        body_movements = []
        emotion_analysis = {"analysis": "Not analyzed", "emotions": {}}
        hr_analysis_json = {"summary": "Not analyzed", "report": []}

        # Extract audio from video
        video = VideoFileClip(video_path)
        audio = video.audio
        audio.write_audiofile(str(audio_path), codec='pcm_s16le', fps=16000)
        audio.close()
        video.close()

        # Resample audio for speech recognition
        data, samplerate = sf.read(str(audio_path))
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
        if samplerate != 16000:
            num_samples = int(len(data) * 16000 / samplerate)
            data = resample(data, num_samples)
        sf.write(str(audio_path), data, 16000, subtype='PCM_16')

        # Perform speech recognition
        if not os.path.exists("vosk-model-small-en-us-0.15"):
            transcript_text = "Vosk model not found. Download from https://alphacephei.com/vosk/models"
        else:
            model = Model("vosk-model-small-en-us-0.15")
            wf = wave.open(str(audio_path), "rb")
            rec = KaldiRecognizer(model, wf.getframerate())
            transcript_text = ""
            while True:
                data = wf.readframes(4000)
                if not data:
                    break
                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result())
                    transcript_text += result.get("text", "") + " "
            transcript_text += json.loads(rec.FinalResult()).get("text", "")
            wf.close()

        # Analyze body movement
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        prev_frame = None
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % 10 == 0:  # Process every 10th frame
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if prev_frame is not None:
                    frame_diff = cv2.absdiff(prev_frame, gray_frame)
                    motion_score = np.sum(frame_diff) / (frame_diff.shape[0] * frame_diff.shape[1])
                    if motion_score > 5.0:
                        body_movements.append("Movement detected")
                prev_frame = gray_frame.copy()
        cap.release()

        body_movement = "Detected active gestures" if body_movements else "Minimal movement detected"

        # Analyze emotions using OpenAI API
        if transcript_text.strip() and OPENAI_API_KEY:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Analyze the emotion in this interview transcript. Provide a detailed analysis and score the following emotions on a scale of 1-5 (where 5 is strongest): Confidence, Enthusiasm, Authenticity, Nervousness, Professionalism, Positivity. Format your response as JSON with an 'analysis' field containing your text analysis and an 'emotions' object containing each emotion as a key with its score as the value."},
                    {"role": "user", "content": transcript_text}
                ]
            )
            emotion_analysis = json.loads(response.choices[0].message.content.strip())
            
            # Determine dominant emotion
            if emotion_analysis.get("emotions", {}):
                dominant_emotion = max(emotion_analysis["emotions"].items(), key=lambda x: float(x[1]))[0]

        # HR Analysis using OpenAI API
        behavioral_analyst_prompt = f"""
        # Role
        Act as a behavioral analyst specializing in evaluating video transcriptions from blue-collar workers.
        # Goal
        Assess video transcriptions to evaluate specific behavioral metrics, providing a concise summary followed by a comprehensive tabular report.
        # Context
        ## videoTranscription
        {transcript_text}
        ## workerRole
        {params.worker_role}
        ## evaluationMetrics
        {params.evaluation_metrics}
        ## summaryLength
        {params.summary_length}
        ## reportFormat
        {params.report_format}
        # Format
        Return your response in JSON with two fields:
        1. "summary": A concise textual summary of the evaluation findings ({params.summary_length} length).
        2. "report": A table in the form of a list of dictionaries, where each dictionary has:
           - "metric": The name of the behavioral metric.
           - "score": A score from 1-5.
           - "comments": A brief textual explanation of the score.
        """
        if transcript_text.strip() and OPENAI_API_KEY:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": behavioral_analyst_prompt},
                    {"role": "user", "content": transcript_text}
                ]
            )
            hr_analysis_json = json.loads(response.choices[0].message.content.strip())

        # Generate HTML report
        html_report = generate_html_report(
            transcript_text,
            dominant_emotion,
            body_movement,
            emotion_analysis,
            hr_analysis_json
        )
        html_report_path = RESULTS_DIR / f"{result_id}.html"
        with open(html_report_path, "w", encoding="utf-8") as f:
            f.write(html_report)

        # Generate PDF report
        pdf_report_path = RESULTS_DIR / f"{result_id}.pdf"
        try:
            options = {
                'page-size': 'Letter',
                'margin-top': '0.75in',
                'margin-right': '0.75in',
                'margin-bottom': '0.75in',
                'margin-left': '0.75in',
                'encoding': "UTF-8",
                'no-outline': None,
                'enable-local-file-access': None
            }
            pdfkit.from_string(html_report, str(pdf_report_path), options=options)
        except Exception as e:
            print(f"PDF generation failed: {str(e)}")
            pdf_report_path = None

        # Prepare the result
        result = {
            "transcript": transcript_text,
            "physical_analysis": {
                "dominant_emotion": dominant_emotion,
                "body_movement": body_movement
            },
            "emotional_analysis": emotion_analysis,
            "hr_analysis": hr_analysis_json,
            "report_links": {
                "html_report": f"/video/api/download-report/{result_id}?format=html",
                "pdf_report": f"/video/api/download-report/{result_id}?format=pdf" if pdf_report_path and os.path.exists(pdf_report_path) else None
            }
        }

        # Save result to file
        result_file = RESULTS_DIR / f"{result_id}.json"
        with open(result_file, "w") as f:
            json.dump(result, f)

        # Clean up temp files
        try:
            os.remove(str(audio_path))
        except Exception:
            pass

        return result

    except Exception as e:
        error_result = {
            "error": str(e),
            "timestamp": time.time()
        }
        return error_result

def generate_html_report(transcript_text, dominant_emotion, body_movement, emotion_analysis, hr_analysis_json):
    hr_analysis_html = f"""
    <p><strong>Summary:</strong> {hr_analysis_json.get('summary', 'No summary provided.')}</p>
    <table style="width: 100%; border-collapse: collapse; margin-top: 15px;">
        <thead>
            <tr style="background-color: #2c3e50; color: white;">
                <th style="padding: 10px; border: 1px solid #ddd;">Metric</th>
                <th style="padding: 10px; border: 1px solid #ddd;">Score</th>
                <th style="padding: 10px; border: 1px solid #ddd;">Comments</th>
            </tr>
        </thead>
        <tbody>
    """
    for row in hr_analysis_json.get('report', []):
        hr_analysis_html += f"""
            <tr>
                <td style="padding: 10px; border: 1px solid #ddd;">{row.get('metric', 'N/A')}</td>
                <td style="padding: 10px; border: 1px solid #ddd; text-align: center;">{row.get('score', 'N/A')}/5</td>
                <td style="padding: 10px; border: 1px solid #ddd;">{row.get('comments', 'No comments')}</td>
            </tr>
        """
    hr_analysis_html += "</tbody></table>"

    recommendations = "<li>No recommendations available</li>"
    try:
        if OPENAI_API_KEY:
            recommendations_response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Based on this HR analysis of an interview, provide 3-5 concise, bullet-point recommendations for the hiring manager. Format as HTML <li> elements."},
                    {"role": "user", "content": json.dumps(hr_analysis_json)}
                ]
            )
            recommendations = recommendations_response.choices[0].message.content.strip()
    except Exception:
        pass

    html_report = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>HR Interview Analysis Report</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; color: #333; margin: 0; padding: 20px; background-color: #f9f9f9; }}
            .report-container {{ max-width: 800px; margin: 0 auto; background-color: #fff; padding: 30px; box-shadow: 0 0 20px rgba(0, 0, 0, 0.1); border: 1px solid #ddd; border-radius: 8px; }}
            .header {{ text-align: center; margin-bottom: 30px; padding-bottom: 20px; border-bottom: 2px solid #2c3e50; }}
            h1 {{ color: #2c3e50; margin-bottom: 10px; }}
            h2 {{ color: #3498db; margin-top: 25px; padding-bottom: 10px; border-bottom: 1px solid #eee; }}
            .section {{ margin-bottom: 30px; padding: 20px; background-color: #f8f9fa; border-radius: 5px; }}
            .emotion-scores {{ display: flex; flex-wrap: wrap; gap: 15px; margin-top: 15px; }}
            .emotion-score {{ background-color: #e3f2fd; border-radius: 5px; padding: 10px 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); width: calc(33% - 15px); box-sizing: border-box; }}
            .emotion-name {{ font-weight: bold; color: #1565c0; margin-bottom: 5px; }}
            .score-bar {{ background-color: #eceff1; height: 10px; border-radius: 5px; margin-top: 5px; overflow: hidden; }}
            .score-fill {{ height: 100%; background-color: #2196f3; }}
            .transcript {{ max-height: 300px; overflow-y: auto; white-space: pre-wrap; background-color: #f5f5f5; padding: 15px; border-radius: 5px; border-left: 4px solid #3498db; font-family: 'Courier New', Courier, monospace; }}
            .footer {{ text-align: center; margin-top: 30px; padding-top: 20px; border-top: 1px solid #eee; color: #7f8c8d; font-size: 0.9em; }}
            .recommendations {{ background-color: #e8f5e9; padding: 15px; border-radius: 5px; margin-top: 15px; border-left: 4px solid #4caf50; }}
            @media print {{ body {{ background-color: white; padding: 0; }} .report-container {{ box-shadow: none; border: none; max-width: 100%; }} }}
        </style>
    </head>
    <body>
        <div class="report-container">
            <div class="header">
                <h1>HR Interview Analysis Report</h1>
                <p>Generated on {time.strftime("%B %d, %Y at %H:%M")}</p>
            </div>
            <div class="section">
                <h2>1. Physical Analysis</h2>
                <p><strong>Dominant Facial Emotion:</strong> {dominant_emotion}</p>
                <p><strong>Body Movement Analysis:</strong> {body_movement}</p>
            </div>
            <div class="section">
                <h2>2. Emotional Analysis</h2>
                <p>{emotion_analysis.get('analysis', 'Analysis not available')}</p>
                <div class="emotion-scores">
    """
    for emotion, score in emotion_analysis.get('emotions', {}).items():
        try:
            score_int = int(float(score))
        except (ValueError, TypeError):
            score_int = 0
        html_report += f"""
                    <div class="emotion-score">
                        <div class="emotion-name">{emotion}</div>
                        <div class="score-value">{score}/5</div>
                        <div class="score-bar">
                            <div class="score-fill" style="width: {score_int * 20}%;"></div>
                        </div>
                    </div>
        """
    html_report += f"""
                </div>
            </div>
            <div class="section">
                <h2>3. Interview Transcript</h2>
                <div class="transcript">{transcript_text.replace("\n", "<br>")}</div>
            </div>
            <div class="section">
                <h2>4. HR Insights</h2>
                <div class="hr-analysis">{hr_analysis_html}</div>
                <div class="recommendations">
                    <h3>Key Recommendations</h3>
                    <ul>{recommendations}</ul>
                </div>
            </div>
            <div class="footer">
                <p>Generated by HR Video Interview Analysis Tool</p>
            </div>
        </div>
    </body>
    </html>
    """
    return html_report