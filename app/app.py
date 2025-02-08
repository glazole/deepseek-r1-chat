import gradio as gr
import requests
import logging
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# URL Ollama API
OLLAMA_API = "http://127.0.0.1:11434"

# Проверка подключения к Ollama перед запуском
def test_ollama_connection():
    try:
        response = requests.get(f"{OLLAMA_API}/api/tags", timeout=5)
        response.raise_for_status()
        logging.info("✅ Успешное подключение к Ollama")
    except requests.exceptions.RequestException as e:
        logging.error(f"❌ Ошибка подключения к Ollama: {e}")

# Запускаем тестовое подключение
test_ollama_connection()

# Инициализация движка LLM
def get_llm_engine(model_name):
    return ChatOllama(
        model=model_name,
        base_url=OLLAMA_API,
        temperature=0.3
    )

# Настройки системного промпта
SYSTEM_TEMPLATE = """You are an expert AI coding assistant. Provide concise, correct solutions 
with strategic print statements for debugging. Always respond in English."""

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_TEMPLATE),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

class ChatBot:
    def __init__(self):
        self.message_log = [{"role": "ai", "content": "Hi! I'm DeepSeek. How can I help you code today? 💻"}]
        self.chat_history = []

    def generate_ai_response(self, user_input, llm_engine):
        """Генерация ответа от AI"""
        self.chat_history.append(HumanMessage(content=user_input))
        
        logging.info(f"📝 Отправка запроса: {user_input}")

        # Запрос к модели
        chain = chat_prompt | llm_engine | StrOutputParser()
        response = chain.invoke({
            "input": user_input,
            "chat_history": self.chat_history
        })

        logging.info(f"💡 Ответ от модели: {response}")

        self.chat_history.append(AIMessage(content=response))
        return response

    def chat(self, message, model_choice, history):
        """Обработка чата в Gradio"""
        if not message:
            return "", history
        
        logging.debug(f"📩 Входящее сообщение: {message}")
        logging.debug(f"🔄 Выбранная модель: {model_choice}")

        llm_engine = get_llm_engine(model_choice)
        logging.debug("✅ LLM-движок успешно инициализирован")
        
        # Добавляем сообщение пользователя в лог
        self.message_log.append({"role": "user", "content": message})
        
        # Генерация ответа
        ai_response = self.generate_ai_response(message, llm_engine)
        
        # Добавляем ответ AI в лог
        self.message_log.append({"role": "ai", "content": ai_response})
        
        # Обновляем историю сообщений (важно: корректный формат)
        history.append((message, ai_response))
        return "", history

def create_demo():
    chatbot = ChatBot()
    
    with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", neutral_hue="zinc")) as demo:
        gr.Markdown("# 🧠 DeepSeek Code Companion")
        gr.Markdown("🚀 Your AI Pair Programmer with Debugging Superpowers")
        
        with gr.Row():
            with gr.Column(scale=4):
                chatbot_component = gr.Chatbot(
                    value=[("Hello!", "Hi! I'm DeepSeek. How can I help you code today? 💻")],
                    height=500,
                    type="messages"  # Добавлено для совместимости с Gradio 4.x+
                )
                msg = gr.Textbox(
                    placeholder="Type your coding question here...",
                    show_label=False
                )
                
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
            outputs=[msg, chatbot_component]
        )

    return demo

if __name__ == "__main__":
    demo = create_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860)