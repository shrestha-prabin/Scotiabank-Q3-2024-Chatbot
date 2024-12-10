import os
import pickle
import time

import pandas as pd
import requests
from flask import Flask, jsonify, render_template, request

from app.chatbot import ChatbotModel

# model_path = "app/models/bert.pth"

# if not os.path.exists(model_path):
#     with requests.get("https://file.io/HyWEg0b2LMaI", stream=True) as r:
#         r.raise_for_status()
#         with open(model_path, "wb") as f:
#             for chunk in r.iter_content(chunk_size=8192):
#                 f.write(chunk)


def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping()

    model = ChatbotModel()

    @app.route("/", methods=["GET"])
    def home():
        return render_template("home.html")

    @app.route("/message", methods=["POST"])
    def message():
        source = request.json.get("source")
        query = request.json.get("query")

        print(source, query)
        response = model.predict(source, query)
        return jsonify(response)

    return app
