# run.py
import uvicorn
import multiprocessing

def run_app():
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

def run_twilio():
    uvicorn.run("twilio_ws:app", host="0.0.0.0", port=8001, reload=True)

if __name__ == "__main__":
    p1 = multiprocessing.Process(target=run_app)
    p2 = multiprocessing.Process(target=run_twilio)
    p1.start()
    p2.start()
    p1.join()
    p2.join()
