from flask import Flask, request, jsonify
import soundfile as sf
from google.cloud import speech
import torch
import numpy as np
from model import CNNTransformer  # 모델 정의 파일로 분리했다고 가정

# Flask 앱 초기화
app = Flask(__name__)

# Google Speech-to-Text 클라이언트 초기화
client = speech.SpeechClient()

# 딥러닝 모델 로드
model = CNNTransformer(input_size=257, num_classes=1)
model.load_state_dict(torch.load("cnn_transformer_model.pth"))
model.eval()

# 음성 파일 처리 함수
def process_audio_file(file_path):
    audio_data, _ = sf.read(file_path)
    # ...원래 코드에서 모델로 예측하는 부분 추가...
    return {"is_screaming": False}  # 예시 반환값

# Google Speech-to-Text API 함수
def transcribe_audio(file_path):
    with open(file_path, "rb") as audio_file:
        content = audio_file.read()
    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=44100,
        language_code="en-US",
        alternative_language_codes=["ko-KR"]
    )
    response = client.recognize(config=config, audio=audio)
    transcribed_text = " ".join([result.alternatives[0].transcript for result in response.results])
    return transcribed_text

# Flask 엔드포인트
@app.route("/process-audio", methods=["POST"])
def process_audio():
    audio = request.files['audio']
    file_path = "uploaded_audio.wav"
    audio.save(file_path)

    # Google API로 텍스트 변환
    transcription = transcribe_audio(file_path)

    # 딥러닝 모델로 Screaming 여부 판단
    screaming_result = process_audio_file(file_path)

    return jsonify({
        "transcription": transcription,
        "screaming_detected": screaming_result["is_screaming"]
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
