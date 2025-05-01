from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Form
from fastapi.responses import Response, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from twilio.twiml.voice_response import VoiceResponse
from datetime import datetime
from app_transcribe import AudioProcessor, collection
import os, json, base64, uuid
from pydub import AudioSegment

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Twilio transcription backend running"}

@app.post("/voice")
async def voice(request: Request):
    print("[Twilio] /voice endpoint hit")

    response = VoiceResponse()
    response.say("This call will be recorded for transcription.")
    
    response.dial(
        "+17633369510",  # replace with actual number
        record="record-from-answer",
        recording_status_callback="https://speech-transcriber-gtku.onrender.com/recording",
        recording_status_callback_method="POST",
        recording_channels="dual"
    )

    return Response(content=str(response), media_type="application/xml")


# Handle Twilio's recording callback
@app.post("/recording")
async def recording_callback(
    RecordingUrl: str = Form(...),
    RecordingSid: str = Form(...),
    CallSid: str = Form(...)
):
    print(f"[Recording] Callback for call {CallSid}")
    audio_url = f"{RecordingUrl}.mp3"

    try:
        print("[AssemblyAI] Uploading Twilio .mp3 recording...")
        utterances, speaker_map = AudioProcessor.process_with_assemblyai(audio_url, is_url=True)
        full_text = " ".join([u["text"] for u in utterances])
        keywords = AudioProcessor.extract_keywords(full_text, top_n=3)

        result = {
            "source_type": "twilio-recording",
            "timestamp": datetime.now(),
            "utterances": utterances,
            "speaker_mapping": speaker_map,
            "keywords": keywords,
            "audio_file": audio_url
        }

        print("[DB] Inserting into MongoDB...")
        collection.insert_one(result)
        print("[Saved] Transcription complete.")
    except Exception as e:
        print(f"[Recording Error] {e}")
    return {"status": "ok"}

@app.get("/ping")
async def ping():
    return {"status": "ok"}

