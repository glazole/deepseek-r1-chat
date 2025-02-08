import gradio as gr
from langchain_ollama import ChatOllama
import requests
import logging
import json

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# URL Ollama в Docker
OLLAMA_API = "http://ollama:11434"

# Проверка подключения к Ollama
def test_ollama_connection():
    payload = {
        "model": "deepseek-r1:1.5b",
        "messages": [{"role": "system", "content": "You are an expert Python and ML/AI coding assistant"},
                     {"role": "user", "content": "Hello!"}]
    }
    try:
        response = requests.post(f"{OLLAMA_API}/api/chat", json=payload)
        logging.info(f"Raw response from Ollama: {response.text}")  # Логируем ответ
        response.raise_for_status()  # Проверяем HTTP-ошибки

        json_data = response.json()  # Парсим JSON
        logging.info(f"Test response JSON: {json_data}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Error connecting to Ollama: {e}")

# Вызываем тестовое подключение
test_ollama_connection()

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# Функция получения LLM
def get_llm_engine(model_name):
    return ChatOllama(
        model=model_name,
        base_url=OLLAMA_API,
        temperature=0.3,
        stream=True
    )

SYSTEM_TEMPLATE = """You are an expert AI coding assistant. Provide concise, correct solutions with logging.info statements for debugging. Always respond in English."""

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_TEMPLATE),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

class ChatBot:
    def __init__(self):
        self.chat_history = []

    def generate_ai_response(self, user_input, llm_engine):
        self.chat_history.append(HumanMessage(content=user_input))
        try:
            chain = chat_prompt | llm_engine | StrOutputParser()
            response = chain.invoke({
                "input": user_input,
                "chat_history": self.chat_history
            })

            full_response = ""
            for line in response.split("\n"):
                if not line.strip():
                    continue
                try:
                    json_data = json.loads(line)
                    if isinstance(json_data, dict) and "message" in json_data and "content" in json_data["message"]:
                        full_response += json_data["message"]["content"]
                    else:
                        logging.warning(f"⚠️ Unexpected JSON format: {line}")
                except json.JSONDecodeError:
                    logging.warning(f"⚠️ Invalid JSON: {line}")

            self.chat_history.append(AIMessage(content=full_response.strip()))
            return full_response.strip()

        except Exception as e:
            logging.error(f"❌ AI Error: {e}")
            return "Error: Failed to process response from Ollama"

    def chat(self, message, model_choice, history):
        if not message:
            return "", history

        logging.debug(f"User input: {message}")
        logging.debug(f"Selected model: {model_choice}")

        llm_engine = get_llm_engine(model_choice)
        logging.debug("LLM engine initialized")

        ai_response = self.generate_ai_response(message, llm_engine)
        logging.debug(f"AI response: {ai_response}")

        history.append((message, ai_response))  # Форматируем в (str, str) для Gradio
        return "", history

def create_demo():
    chatbot = ChatBot()

    with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", neutral_hue="zinc")) as demo:
        gr.Markdown("# 🧠 DeepSeek Code Companion")
        gr.Markdown("🚀 Your AI Pair Programmer with Debugging Superpowers")

        with gr.Row():
            with gr.Column(scale=4):
                chatbot_component = gr.Chatbot(
                    value=[("Hello!", "Hi! I'm DeepSeek. How can I help you code today? 💻")],  # Формат исправлен
                    height=500,
                )
                msg = gr.Textbox(placeholder="Type your coding question here...", show_label=False)

            with gr.Column(scale=1):
                model_dropdown = gr.Dropdown(
                    choices=["deepseek-r1:1.5b", "deepseek-r1:7b"],
                    value="deepseek-r1:1.5b",
                    label="Choose Model"
                )

                gr.Markdown("### Model Capabilities")
                gr.Markdown("""
                - 🐍 Python Expert
                - 🐞 Debugging Assistant
                - 📝 Code Documentation
                - 💡 Solution Design
                """)

                gr.Markdown("Built with [Ollama](https://ollama.ai/) | [LangChain](https://python.langchain.com/)")

        msg.submit(
            fn=chatbot.chat,
            inputs=[msg, model_dropdown, chatbot_component],
            outputs=[chatbot_component]
        ).then(lambda: "", None, msg)

    return demo

if __name__ == "__main__":
    demo = create_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860)