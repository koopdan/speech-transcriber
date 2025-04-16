#!/bin/bash

# Create folders if they don't exist
mkdir -p recorded_audio uploads

# Run the appropriate FastAPI app
uvicorn twilio_ws:app --host 0.0.0.0 --port 10000
