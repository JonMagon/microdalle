import json
import os
import random
import string
import logging
from pathlib import Path
from threading import Thread
from typing import Optional
from datetime import datetime

import requests
from flask import Flask, request, send_file, jsonify
from openai import OpenAI
from openai import OpenAIError

import boto3

from io import BytesIO

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def generate_save_file_name() -> str:
    current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    return f"{current_time}-{''.join(random.choices(string.ascii_letters, k=3))}.png"


def api_key() -> Optional[str]:
    if os.path.isfile("key"):
        with open("key", encoding="utf=8") as fd:
            return fd.readline().strip()
    return None


def upload_file_to_s3(file_name: str, metadata: dict, url: str):
    s3_bucket_name = os.environ.get("S3_BUCKET_NAME")
    aws_access_key_id = os.environ.get("S3_ACCESS_KEY_ID")
    aws_secret_access_key = os.environ.get("S3_SECRET_ACCESS_KEY")
    s3_endpoint_url = os.environ.get("S3_ENDPOINT")

    s3_client = boto3.client(
        's3',
        endpoint_url=s3_endpoint_url,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )

    log.info("Uploading file to S3 bucket %s", s3_bucket_name)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with BytesIO() as buffer:
            # Читаем содержимое файла по частям
            for chunk in r.iter_content():
                if chunk:  # фильтр для удаления keep-alive новых чанков
                    buffer.write(chunk)
            buffer.seek(0)
            s3_client.upload_fileobj(buffer, s3_bucket_name, file_name)

    # Загружаем метаданные
    metadata_key = file_name.replace('.png', '.json')
    s3_client.put_object(
        Bucket=s3_bucket_name,
        Key=metadata_key,
        Body=json.dumps(metadata, indent=4).encode('utf-8')
    )

    log.info("File and metadata uploaded successfully to %s/%s", s3_bucket_name, file_name)


client = OpenAI(api_key=api_key()) # base_url="https://api.proxyapi.ru/openai/v1"

app = Flask(__name__)


@app.route("/generate", methods=["POST"])
def generate():
    # Get the text from the user
    prompt = request.json["prompt"]
    resolution = request.json["resolution"]
    model_name = request.json["model"]
    log.info("Sending request to OpenAI with prompt: %s", prompt)

    if model_name == "dall-e-3-hd":
        model = "dall-e-3"
        quality = "hd"
    else:
        model = model_name
        quality = "standard"

    try:
        response = client.images.generate(
            model=model,
            prompt=prompt,
            size=resolution,
            quality=quality,
            n=1,
        )
    except OpenAIError as e:
        return e.message, 500

    log.info("Response received: %s", response)
    url = response.data[0].url

    # Fire and Forget, It will download the file in the background. It is not production-ready, but that should
    # be enough for now
    metadata = {**response.data[0].model_dump(), "prompt":prompt}
    Thread(target=upload_file_to_s3, args=(generate_save_file_name(), metadata, url)).start()

    return jsonify(response.data[0].model_dump())


@app.route("/", methods=["GET"])
def index():
    return send_file("index.html", mimetype="text/html")

@app.route("/balance", methods=["GET"])
def balance():
    url = 'https://'

    headers = {
        'Authorization': 'Bearer '
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        try:
            balance = response.json().get("balance")
            return jsonify({'balance': balance})
        except ValueError:
            return b'Invalid response from external API', 500
    else:
        return response.content, response.status_code

@app.route("/inprogress.gif", methods=["GET"])
def inprogress():
    return send_file("inprogress.gif", mimetype="image/gif")


if __name__ == "__main__":
    app.run(debug=True)
