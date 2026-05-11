from flask import Flask

app = Flask(__name__)

@app.route("/")
def home():
    return "AI Motion Battle Arena is running!"


import cv2
import mediapipe as mp
import math
import numpy as np
