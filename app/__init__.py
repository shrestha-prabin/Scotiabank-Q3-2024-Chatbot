import pickle

import pandas as pd
from flask import Flask, render_template, request, jsonify
import time
from app.chatbot import ChatbotModel


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
        query = request.json.get('query')
        print(query)
        response = model.predict(query)
        return jsonify(response)

    return app
