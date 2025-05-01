import os
import time
import requests
from pymongo import MongoClient
from keybert import KeyBERT
from requests.auth import HTTPBasicAuth

# Load secrets from environment variables
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
TWILIO_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")

# Setup MongoDB connection
client = MongoClient(MONGO_URI)
db = client["Speech-to-text"]
collection = db["voice"]

class AudioProcessor:
    @staticmethod
    def process_with_assemblyai(input_data, is_url=False):
        if is_url:
            print("[AssemblyAI] Downloading Twilio recording via authenticated request...")
            response = requests.get(input_data, auth=HTTPBasicAuth(TWILIO_SID, TWILIO_AUTH_TOKEN))

            print("[Twilio Download] Status Code:", response.status_code)
            if response.status_code != 200:
                print("[Twilio Download] Response Text:", response.text)
                raise Exception("Failed to download Twilio recording")

            with open("temp_recording.mp3", "wb") as f:
                f.write(response.content)
            audio_url = AudioProcessor.upload_to_assemblyai("temp_recording.mp3")
        else:
            audio_url = AudioProcessor.upload_to_assemblyai(input_data)

        transcript_id = AudioProcessor.submit_for_transcription(audio_url)
        return AudioProcessor.poll_and_process(transcript_id)

    @staticmethod
    def upload_to_assemblyai(file_path):
        headers = {"authorization": ASSEMBLYAI_API_KEY}
        with open(file_path, 'rb') as f:
            print("[AssemblyAI] Uploading audio file...")
            response = requests.post("https://api.assemblyai.com/v2/upload", headers=headers, data=f)

        print("[Upload Response]", response.status_code, response.text)
        if response.status_code != 200:
            raise Exception("Upload failed")

        return response.json()["upload_url"]

    @staticmethod
    def submit_for_transcription(audio_url):
        json_data = {
            "audio_url": audio_url,
            "speaker_labels": False,
            "dual_channel": True,
            "punctuate": True
        }
        headers = {
            "authorization": ASSEMBLYAI_API_KEY,
            "content-type": "application/json"
        }
        print("[AssemblyAI] Requesting transcription...")
        response = requests.post("https://api.assemblyai.com/v2/transcript", json=json_data, headers=headers)

        print("[Transcript Response]", response.status_code, response.text)
        if response.status_code != 200:
            raise Exception("Transcription request failed")

        return response.json()["id"]

    @staticmethod
    def poll_and_process(transcript_id):
        headers = {"authorization": ASSEMBLYAI_API_KEY}
        status_url = f"https://api.assemblyai.com/v2/transcript/{transcript_id}"

        print("[Polling] Waiting for transcription to complete...")

        while True:
            res = requests.get(status_url, headers=headers)
            data = res.json()

            if data["status"] == "completed":
                print("[Transcription complete ]")
                return data.get("utterances", []), data.get("speakers", [])

            if data["status"] == "error":
                raise Exception(f"Transcription failed: {data['error']}")

            time.sleep(2)

    @staticmethod
    def extract_keywords(text, top_n=3):
        kw_model = KeyBERT()
        return kw_model.extract_keywords(text, top_n=top_n)

