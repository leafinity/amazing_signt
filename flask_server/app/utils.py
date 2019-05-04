#!/usr/bin/env python
from flask import make_response, jsonify, send_file


image_format = {
    'jpg': 'jpeg',
    'png': 'png',
}

def send_file(image_path, attachment_filename)
    response = make_response(send_file(image_path, attachment_filename=attachment_filename))
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response


def send_json_response(custom_dict, error_code=200):
    respose_dict = {
        'success': True
    }
    respose_dict.update(custom_dict)
    response = make_response(jsonify(respose_dict), error_code)
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response


def send_error_response(msg, error_code=500):
    print('fail', msg)
    return send_json_response({ 
        'success': False,
        'message': msg,
    }, error_code=error_code)


def get_file_extension(filename):
    return filename.rsplit('.', 1)[1].lower()


def allowed_file(filename, allow_extensions):
    if '.' in filename:
        return get_file_extension(filename) in allow_extensions
    else:
        return False

