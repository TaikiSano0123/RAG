from flask import Flask, render_template, request, jsonify
import google.generativeai as genai

app = Flask(__name__)

@app.route('/test')
def test():
    return "Test route is working!"

# Gemini APIの設定
genai.configure(api_key='AIzaSyAk71sNdS7VRg96eCHflHULLaHeDDI0h9E')
model = genai.GenerativeModel('gemini-pro')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json['message']
    response = model.generate_content(user_message)
    return jsonify({'response': response.text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)