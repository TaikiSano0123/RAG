from flask import Flask, request, jsonify, render_template
from google.cloud import aiplatform

app = Flask(__name__)

PROJECT_ID = "instance-20240723-013646"  # あなたのプロジェクトIDに置き換えてください
LOCATION = "us-central1-f"  # 適切なロケーションに変更してください

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data['message']

    # Gemini API を呼び出し
    aiplatform.init(project=PROJECT_ID, location=LOCATION)
    model = aiplatform.Model(f"projects/{PROJECT_ID}/locations/{LOCATION}/models/gemini-pro")
    response = model.predict(instances=[{"content": user_input}])

    return jsonify({"response": response.predictions[0]['content']})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)