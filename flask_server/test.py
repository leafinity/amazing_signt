import time
import json
import requests
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--hostname', type=str, default='http://87a0565e.ngrok.io/')
args = parser.parse_args()

r = requests.post(args.hostname + 'generate_scene/%s' % 'mountain')
if r.status_code == 200:
    with open('test_recieved.jpg', 'wb') as f:
        f.write(r.content)
else:
    print(r.status_code)

