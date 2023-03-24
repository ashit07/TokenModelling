from flask import Flask, render_template, request, url_for, jsonify
from FeatureExtractUtil import generateFeatureTopics
import json
app = Flask(__name__)

@app.route('/')
def get_data(data=None):
    return "Hello"

@app.route('/features', methods=["POST"])
def get_dataFeatures():
    data = request.get_json(force=True) 
    print("Input Json is: ")
    print(data)

    list_data = generateFeatureTopics(data=data)
    return json.dumps(list_data)

#get_data()