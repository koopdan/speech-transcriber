from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from twilio.twiml.voice_response import VoiceResponse
import wave, uuid, os, base64, json
from datetime import datetime

from app import AudioProcessor, collection  # Reuse your logic

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CALL_AUDIO_DIR = "recorded_audio"
os.makedirs(CALL_AUDIO_DIR, exist_ok=True)

@app.post("/voice")
async def voice(request: Request):
    response = VoiceResponse()
    response.say("Welcome to the transcription service. Connecting your call now.")
    response.start().stream(url="wss://speech-to-text-xu54.onrender.com/ws/transcription")
    response.dial("13203396951")  # Replace with actual number
    return Response(content=str(response), media_type="application/xml")

@app.websocket("/ws/transcription")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("[WebSocket Connected]")
    session_id = str(uuid.uuid4())
    raw_audio = b""

    try:
        while True:
            data = await websocket.receive_text()
            json_data = json.loads(data)
            if json_data.get("event") == "media":
                payload = json_data["media"]["payload"]
                raw_audio += base64.b64decode(payload)
    except WebSocketDisconnect:
        print("[WebSocket Disconnected]")
        audio_path = os.path.join(CALL_AUDIO_DIR, f"{session_id}.wav")
        with wave.open(audio_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(8000)
            wf.writeframes(raw_audio)

        try:
            utterances, speaker_names = AudioProcessor.process_with_assemblyai(audio_path)
            full_text = " ".join([u["text"] for u in utterances])
            keywords = AudioProcessor.extract_keywords(full_text, top_n=3)

            result = {
                "source_type": "twilio-call",
                "timestamp": datetime.now(),
                "utterances": utterances,
                "speaker_mapping": speaker_names,
                "keywords": keywords,
                "audio_file": audio_path
            }

            collection.insert_one(result)
            print("[Saved to DB]")

        except Exception as e:
            print(f"[Transcription Error] {e}")
