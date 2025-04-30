from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from twilio.twiml.voice_response import VoiceResponse
from datetime import datetime
from app_transcribe import AudioProcessor, collection
from pydub import AudioSegment
import uuid, os, json, base64

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

@app.get("/ping")
async def ping():
    return {"status": "ok"}

@app.post("/voice")
async def voice(request: Request):
    print("[Twilio] /voice endpoint hit")

    response = VoiceResponse()
    response.say("Connecting your call now.")
    response.pause(length=2.5)  # Gives time for WebSocket to be ready
    response.start().stream(
        url="wss://speech-transcriber-gtku.onrender.com/ws/transcription"
    )
    response.dial("+17633369510")
    return Response(content=str(response), media_type="application/xml")

@app.websocket("/ws/transcription")
async def websocket_endpoint(websocket: WebSocket):
    print("[WebSocket] waiting for accept...")
    await websocket.accept()
    print("[WebSocket] Connected")

    audio_buffer = b""
    session_id = str(uuid.uuid4())

    try:
        while True:
            msg = await websocket.receive_text()
            data = json.loads(msg)
            if data.get("event") == "media":
                audio_buffer += base64.b64decode(data["media"]["payload"])
    except WebSocketDisconnect:
        print("[WebSocket] Disconnected")
        print(f"[Buffer Size] Received {len(audio_buffer)} bytes")

        os.makedirs("recorded_audio", exist_ok=True)
        file_path = f"recorded_audio/{session_id}.wav"

        try:
            print("[Audio Export] Starting export...")
            audio_segment = AudioSegment(
                data=audio_buffer,
                sample_width=1,
                frame_rate=8000,
                channels=1
            )
            audio_segment.export(file_path, format="wav")
            print(f"[Audio saved] {file_path}")
        except Exception as e:
            print(f"[Audio Save Error] {e}")
            return

        try:
            print("[Uploading to AssemblyAI...]")
            utterances, speaker_map = AudioProcessor.process_with_assemblyai(file_path)

            # Handle missing utterances gracefully
            if not utterances:
                print("[Warning] No utterances returned — fallback to plain text.")
                full_text = AudioProcessor.get_raw_text(file_path)
            else:
                full_text = " ".join([u["text"] for u in utterances])

            keywords = AudioProcessor.extract_keywords(full_text, top_n=3)

            result = {
                "source_type": "twilio-call",
                "timestamp": datetime.now(),
                "utterances": utterances,
                "speaker_mapping": list(speaker_map),
                "keywords": keywords,
                "audio_file": file_path
            }

            print("[DB] Attempting to insert:", result)
            collection.insert_one(result)
            print("[Saved] Transcription complete.")

        except Exception as e:
            print(f"[Transcription Error] {e}")

