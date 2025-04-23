import os
import requests
from pymongo import MongoClient
from keybert import KeyBERT

ASSEMBLYAI_API_KEY = os.getenv("ada3df25b909471ca405dd86fc221940")
MONGO_URI = os.getenv("mongodb://localhost:27017/") or "mongodb://localhost:27017"

client = MongoClient(MONGO_URI)
db = client["speech"]
collection = db["voice"]

class AudioProcessor:
    @staticmethod
    def process_with_assemblyai(file_path):
        upload_url = AudioProcessor.upload_to_assemblyai(file_path)
        transcript_url = AudioProcessor.submit_for_transcription(upload_url)
        return AudioProcessor.poll_and_process(transcript_url)

    @staticmethod
    def upload_to_assemblyai(file_path):
        headers = {"authorization": ASSEMBLYAI_API_KEY}
        with open(file_path, 'rb') as f:
            response = requests.post("https://api.assemblyai.com/v2/upload", headers=headers, files={"file": f})
        return response.json()["upload_url"]

    @staticmethod
    def submit_for_transcription(audio_url):
        json_data = {
            "audio_url": audio_url,
            "speaker_labels": True,
            "dual_channel": False,
            "punctuate": True
        }
        headers = {"authorization": ASSEMBLYAI_API_KEY, "content-type": "application/json"}
        response = requests.post("https://api.assemblyai.com/v2/transcript", json=json_data, headers=headers)
        return response.json()["id"]

    @staticmethod
    def poll_and_process(transcript_id):
        headers = {"authorization": ASSEMBLYAI_API_KEY}
        status_url = f"https://api.assemblyai.com/v2/transcript/{transcript_id}"
        while True:
            res = requests.get(status_url, headers=headers)
            data = res.json()
            if data["status"] == "completed":
                return data["utterances"], data["speakers"] if "speakers" in data else {}
            elif data["status"] == "error":
                raise Exception(f"Transcription failed: {data['error']}")

    @staticmethod
    def extract_keywords(text, top_n=3):
        kw_model = KeyBERT()
        return kw_model.extract_keywords(text, top_n=top_n)

