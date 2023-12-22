import os
from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import requests
from io import BytesIO

app = Flask(__name__)

def allowed_file(filename):
    return "." in filename and \
        filename.split(".", 1)[1] in app.config["ALLOWED_EXTENSIONS"]

model = load_model("model.h5", compile=False)
with open("labels.txt", "r") as file:
    labels = file.read().splitlines()
with open("calories.txt", "r") as file:
    calories = file.read().splitlines()
with open("benefit.txt", "r") as file:
    benefits = [line.split('\n') for line in file.read().split('\n\n')]

@app.route("/")
def index():
    return jsonify({
        "status": {
            "code": 200,
            "message": "Success fetching the API",
        },
        "data": None
    }), 200

@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    if request.method == "POST":
        image_url = request.json.get("image_url")
        if image_url:
            response = requests.get(image_url)
            if response.status_code == 200:
                # Save input image
                img = Image.open(BytesIO(response.content)).convert("RGB")
                img = img.resize((224, 224))
                img_array = np.asarray(img)
                img_array = np.expand_dims(img_array, axis=0)
                normalized_image_array = (img_array.astype(np.float32) / 127.5) - 1
                data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
                data[0] = normalized_image_array

                # Predicting the image
                prediction = model.predict(data)
                index =  np.argmax(prediction)
                class_names = labels[index]
                confidence_score = prediction[0][index]
                calorie_value = calories[index]
                benefit = benefits[index]

                return jsonify({
                    "status": {
                        "code": 200,
                        "message": "Success predicting",
                    },
                    "data": {
                        "vegetable_prediction": class_names,
                        "confidence": float(confidence_score),
                        "calories": f"{calorie_value} (100 grams)",
                        "benefit": benefit,
                    }
                }), 200
            else:
                return jsonify({
                    "status": {
                        "code": 400,
                        "message": "Client side error"
                    },
                    "data": None
                }), 400
        else:
                return jsonify({
                    "status": {
                        "code": 400,
                        "message": "Invalid image URL"
                    },
                    "data": None
                }), 400
    else:
        return jsonify({
            "status":{
                "code": 405,
                "message": "Method not allowed"
            },
            "data": None,
        }), 405 

if __name__ == "__main__":
    app.run()