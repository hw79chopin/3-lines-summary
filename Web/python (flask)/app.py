import sys
import json
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return "Flask server"

@app.route('/summary', methods=['POST'])
def recommend():
    args = request.get_json(force=True)
    user_input = args.get('user_input', [])

    return jsonify(
        summary = ['이것이 첫 번째 문장', '이것은 두 번째 문장', '이것이 세 번째 문장'],
        userInput = user_input
    )
if __name__ == "__main__":
    app.run(host='localhost', port=5000)