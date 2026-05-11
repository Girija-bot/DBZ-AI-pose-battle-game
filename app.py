from flask import Flask

app = Flask(__name__)

@app.route("/")
def home():
    return """
    <h1>AI Motion Battle Arena</h1>
    <p>Real-time AI pose recognition game built with:</p>

    <ul>
        <li>OpenCV</li>
        <li>MediaPipe</li>
        <li>LangChain</li>
        <li>Claude AI</li>
        <li>Python</li>
    </ul>

    <p>Gameplay demo coming soon.</p>
    """
