from flask import Flask

app = Flask(__name__)

@app.route("/")
def home():
    return """
    <h1>AI Motion Battle Arena</h1>
    <p>AI-powered Dragon Ball Z pose battle game.</p>
    <p>Built with MediaPipe, LangChain, Claude AI, OpenCV, and RAG.</p>
    """
