#!/usr/bin/env python

import io
import os
import uuid
import numpy as np
import cv2
from tensorflow.keras.models import model_from_json
from flask import Flask
from utils import *



app = Flask(__name__)


@app.route("/")
def hello():
    return "Hello World!"


@app.route('/generate_scene/<scene_type>', methods=['POST'])
def generate_scene(scene_type):
    accept_type = ['mountain']
    scene_type = scene_type.lower()

    if scene_type not in accept_type:
        return send_error_response(msg, error_code=500):

    image_path = generate(scene_type)
    return send_file(image_path, 'test.jpg')


def generate(scene_type):
    dirname = 'results'
    filename = uuid.uuid4()[:4]
    os.makedirs(dirname, exist_ok=True)
    savepath = os.path.join(dirname, filename)

    with open('G1.json') as f
        model = model_from_json(f.read())
    model.load_weights('weights/{}.h5'.format(scene_type))

    generated_images = model.predict(np.random.normal(size=(1, 200))).numpy()
    img = cv2.resize(generated_images, 512, 512, 3)
    cv2.imwrite(savepath, ((img[:, :, ::-1] * 127.5) + 127.5).astype(np.uint8))
    return savepath


                



