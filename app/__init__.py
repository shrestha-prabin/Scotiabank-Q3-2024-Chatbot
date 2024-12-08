import pickle

import pandas as pd
from flask import Flask, render_template, request, jsonify
import time

class ChatbotModel:

    def __init__(self):
        # self.load_model()
        pass

    def load_model(self):
        with open("app/models/best.pkl", "rb") as f:
            self.model = pickle.load(f)

    def get_response(self, query):
        time.sleep(1)
        return "Officia irure culpa duis reprehenderit Lorem esse officia ipsum et consectetur officia. Duis Lorem veniam excepteur anim laborum laboris dolor ullamco adipisicing mollit laborum deserunt. Enim exercitation aliqua deserunt adipisicing nisi. Culpa cupidatat ad nostrud nisi aliqua. In dolore culpa cillum culpa adipisicing consectetur ullamco labore est mollit sit velit sunt." + query



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
        response = model.get_response(query)
        return jsonify(response)

    return app
