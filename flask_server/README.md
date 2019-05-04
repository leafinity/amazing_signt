# Flask Server on Jupyter hub

## Execution

start flask server

    export FLASK_APP=./app/main.py
    PYTHONPATH=./app python -m flask run

start ngrok

    ./ngrok http 5000


## How to Use

* start server

* post josn object to "/generate_scene/<scene_type>"

## Else

if you cannot start ngrok, create an ngrok account and follow the tutorial.

## Model and Weights
https://drive.google.com/open?id=1DppoNi9fswsGfwOQbDiYiiKjr1DnuVAX

put G1.json under `model`
put *.h5 under `weights`