from flask import Flask, render_template, request, jsonify
from langchain.llms.base import LLM
from langchain.agents import initialize_agent, Tool
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import google.generativeai as genai
from typing import Optional, List

app = Flask(__name__)

# Gemini APIの設定
genai.configure(api_key='AIzaSyAk71sNdS7VRg96eCHflHULLaHeDDI0h9E')
model = genai.GenerativeModel('gemini-pro')

class CustomGeminiLLM(LLM):
    model_name: str = "gemini-pro"
    temperature: float = 0.7

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = model.generate_content(prompt)
        return response.text

    @property
    def _llm_type(self) -> str:
        return "custom_gemini"

llm = CustomGeminiLLM()

# メモリの初期化
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# ディベート設定ツール
debate_setup_prompt = PromptTemplate(
    input_variables=["topic"],
    template="ディベートのトピック '{topic}' について、どのような議論をAIにさせるべきか提案してください。具体的な論点や、賛成側・反対側の立場を示してください。"
)
debate_setup_chain = LLMChain(llm=llm, prompt=debate_setup_prompt)

# ディベート実行ツール
debate_execute_prompt = PromptTemplate(
    input_variables=["topic", "setup"],
    template="""
トピック '{topic}' について、以下の設定に基づいてディベートを行ってください：

{setup}

以下の形式で厳密にディベートを行い、結果を提示してください。各発言は100〜150字程度とし、合計3ターンのやり取りを行ってください：

モデレーター: ディベートを開始します。テーマは「{topic}」です。まず、賛成側から主張をお願いします。

賛成側: [賛成側の主張]

モデレーター: ありがとうございます。次に、反対側からの反論をお願いします。

反対側: [反対側の反論]

モデレーター: 賛成側、反対側の意見を踏まえてさらなる主張をお願いします。

賛成側: [賛成側の再反論]

モデレーター: 反対側、いかがでしょうか。

反対側: [反対側の再反論]

モデレーター: 最後に、両者から最終主張をお願いします。まず賛成側からどうぞ。

賛成側: [賛成側の最終主張]

モデレーター: 続いて反対側、お願いします。

反対側: [反対側の最終主張]

モデレーター: ありがとうございました。これでディベートを終了します。評価者からの評価をお願いします。

評価者: [ディベートの評価（200〜300字程度）。両者の主張を比較し、論理性、説得力、証拠の使用などの観点から分析してください。]

勝者: [賛成 or 反対]

評価理由: [勝者を選んだ理由を100字程度で説明してください。]
"""
)
debate_execute_chain = LLMChain(llm=llm, prompt=debate_execute_prompt)

# ツールの定義
tools = [
    Tool(
        name="DebateSetup",
        func=debate_setup_chain.run,
        description="ディベートの設定を提案するツール"
    ),
    Tool(
        name="DebateExecute",
        func=lambda **kwargs: debate_execute_chain.run(**kwargs),
        description="提案された設定に基づいてディベートを実行するツール。入力: topic, setup"
    )
]

# エージェントの初期化
agent = initialize_agent(tools, llm, agent="chat-conversational-react-description", memory=memory, verbose=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json['message']
    response = agent.run(user_message)
    return jsonify({'response': response})

@app.route('/debate_setup', methods=['POST'])
def debate_setup():
    topic = request.json['topic']
    response = agent.run(f"トピック '{topic}' についてのディベート設定を提案してください。")
    return jsonify({'response': response})

@app.route('/debate_execute', methods=['POST'])
def debate_execute():
    topic = request.json['topic']
    setup = request.json['setup']
    response = agent.run(f"DebateExecute: topic='{topic}', setup='{setup}'")
    return jsonify({'response': response})

@app.errorhandler(Exception)
def handle_exception(e):
    app.logger.error(f"An error occurred: {str(e)}")
    return jsonify(error=str(e)), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)