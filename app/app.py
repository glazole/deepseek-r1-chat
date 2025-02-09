import gradio as gr
import requests
import logging
import time
import re
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# URL Ollama API
OLLAMA_API = "http://ollama:11434"

# Проверка подключения к Ollama перед запуском
def test_ollama_connection(retries=3, delay=3):
    for i in range(retries):
        try:
            response = requests.get(f"{OLLAMA_API}/api/tags", timeout=5)
            response.raise_for_status()
            logging.info("✅ Успешное подключение к Ollama")
            return True
        except requests.exceptions.RequestException as e:
            logging.warning(f"❌ Ошибка подключения ({i+1}/{retries}): {e}")
            time.sleep(delay)
    logging.error("🚨 Не удалось подключиться к Ollama после нескольких попыток")
    return False

if not test_ollama_connection():
    exit(1)

# Инициализация движка LLM
def get_llm_engine(model_name):
    logging.info(f"🚀 Инициализация модели {model_name}")
    try:
        return ChatOllama(model=model_name, base_url=OLLAMA_API, temperature=0.3)
    except Exception as e:
        logging.error(f"❌ Ошибка инициализации модели {model_name}: {e}")
        return None

# Настройки системного промпта
SYSTEM_TEMPLATE = """You are an expert AI coding assistant. Provide concise, correct solutions 
with strategic print statements for debugging. Always respond in English using Markdown formatting for better readability."""

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_TEMPLATE),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

# Класс чат-бота
class ChatBot:
    def __init__(self):
        self.chat_history = [
            # {"role": "assistant", "content": "Hi! I'm **DeepSeek**. How can I help you code today? 💻"}
        ]
    def clean_response(self, response):
        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
        return response
    # Генерация ответа модели на основе введенного пользователем сообщения
    def generate_ai_response(self, user_input, llm_engine):
        logging.info(f"📝 Отправка запроса в модель: {user_input}")
        # Преобразуем историю в список строк, чтобы LangChain не ломался
        formatted_history = [
            message.content if isinstance(message, (HumanMessage, AIMessage)) else message
            for message in self.chat_history
        ]
        self.chat_history.append(HumanMessage(content=user_input))
        chain = chat_prompt | llm_engine | StrOutputParser()
        response = chain.invoke({"input": user_input, "chat_history": formatted_history}) or "⚠️ Ошибка: модель не вернула ответ."
        response = self.clean_response(response)  # Очищаем от <think>...</think>
        self.chat_history.append(AIMessage(content=response))
        logging.info(f"💡 Полный ответ от модели: {response}")
        return response
    # Отработка чата в интерфейсе Gradio
    def chat(self, message, model_choice, history):
        if not message:
            return "", history
        llm_engine = get_llm_engine(model_choice)
        ai_response = self.generate_ai_response(message, llm_engine)
        # Добавляем сообщения в историю
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": ai_response})
        logging.info(f"📜 Обновленная история чата: {history}")
        return "", history
    # Для тестирование чата
    def chat_test(self, message, model_choice, history):
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": "This is a test response."})
        return "", history

        

# Создание интерфейса Gradio
def create_demo():
    chatbot = ChatBot()

    with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", neutral_hue="zinc")) as demo:
        gr.Markdown("# 🧠 DeepSeek Code Companion")
        gr.Markdown("🚀 Your AI Pair Programmer with Debugging Superpowers")
            
        with gr.Row():
            with gr.Column(scale=4):
                chatbot_component = gr.Chatbot(
                    value=[{"role": "assistant", "content": "Hi! I'm **DeepSeek**. How can I help you code today? 💻"}],  
                    show_copy_button=True,
                    height=500,
                    type="messages",
                    render_markdown=True,
                )
                msg = gr.Textbox(placeholder="Type your coding question here...", show_label=False)

                # Кнопка очистки чата
                clear = gr.ClearButton([msg, chatbot_component])

            with gr.Column(scale=1):
                model_dropdown = gr.Dropdown(
                    choices=["deepseek-r1:1.5b", "deepseek-r1:7b", "deepseek-r1:14b"],
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

        # Обрабатываем ввод сообщений
        msg.submit(chatbot.chat, [msg, model_dropdown, chatbot_component], [msg, chatbot_component])

    return demo

if __name__ == "__main__":
    demo = create_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860)
