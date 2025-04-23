from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from twilio.twiml.voice_response import VoiceResponse
from datetime import datetime
import uuid, wave, os, json, base64

from app_transcribe import AudioProcessor, collection

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
    response.say("Connecting your call now.")
    response.start().stream(
        url="wss://speech-transcriber-gtku.onrender.com/ws/transcription"
    )
    response.dial("+17633369510")
    return Response(content=str(response), media_type="application/xml")


@app.websocket("/ws/transcription")
async def websocket_endpoint(websocket: WebSocket):
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
        os.makedirs("recorded_audio", exist_ok=True)
        file_path = f"recorded_audio/{session_id}.wav"
        with wave.open(file_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(8000)
            wf.writeframes(audio_buffer)

        try:
            utterances, speaker_map = AudioProcessor.process_with_assemblyai(file_path)
            full_text = " ".join([u["text"] for u in utterances])
            keywords = AudioProcessor.extract_keywords(full_text, top_n=3)
            print("[DB] Attempting to insert:", result)
            collection.insert_one({
                "source_type": "twilio-call",
                "timestamp": datetime.now(),
                "utterances": utterances,
                "speaker_mapping": speaker_map,
                "keywords": keywords,
                "audio_file": file_path
            })
            print("[Saved] Transcription complete.")
        except Exception as e:
            print(f"[Transcription Error] {e}")

