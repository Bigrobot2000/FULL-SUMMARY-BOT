from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline

# Initialize FastAPI app
app = FastAPI()

# Enable CORS (Allow frontend to connect)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

@app.post("/summarize/")
async def summarize(text: str = Form(...)):
    summary = summarizer(text, max_length=150, min_length=50, do_sample=False)
    return {"summary": summary[0]["summary_text"]}

# Run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
