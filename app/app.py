import gradio as gr
from langchain_ollama import ChatOllama
import requests
import torch
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Новый URL Ollama в Docker
OLLAMA_API = "http://ollama:11434"  # Исправлено

# Тест Ollama перед запуском Gradio
def test_ollama_connection():
    payload = {
        "model": "deepseek-r1:1.5b",
        "messages": [{"role": "system", "content": "You are an expert Python and ML/AI coding assistant"},
                     {"role": "user", "content": "Hello!"}]
    }
    try:
        response = requests.post(f"{OLLAMA_API}/api/chat", json=payload)  # Исправлено
        logging.info(f"Test response: {response.json()}")
    except Exception as e:
        logging.error(f"Error connecting to Ollama: {e}")

# Вызов тестовой функции
test_ollama_connection()

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# Инициализация LLM
def get_llm_engine(model_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    return ChatOllama(
        model=model_name,
        base_url=OLLAMA_API,  # Исправлено
        temperature=0.3
    )


SYSTEM_TEMPLATE = """You are an expert AI coding assistant. Provide concise, correct solutions 
with strategic logging.info statements for debugging. Always respond in English."""

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
        self.chat_history.append(HumanMessage(content=user_input))
        try:
            chain = chat_prompt | llm_engine | StrOutputParser()
            response = chain.invoke({
                "input": user_input,
                "chat_history": self.chat_history
            })
            self.chat_history.append(AIMessage(content=response))
            return response
        except Exception as e:
            logging.error(f"Error generating AI response: {e}")  # Исправлено
            return "Sorry, I encountered an error while generating the response."

    def chat(self, message, model_choice, history):
        if not message:
            return "", history
        
        logging.debug(f"[DEBUG] User input: {message}")
        logging.debug(f"[DEBUG] Selected model: {model_choice}")

        llm_engine = get_llm_engine(model_choice)
        logging.debug("[DEBUG] LLM engine initialized")
        
        self.message_log.append({"role": "user", "content": message})
        
        ai_response = self.generate_ai_response(message, llm_engine)
        logging.debug(f"[DEBUG] AI response: {ai_response}")
        
        self.message_log.append({"role": "ai", "content": ai_response})
        
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
                    value=[(None, "Hi! I'm DeepSeek. How can I help you code today? 💻")],
                    height=500,
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
            outputs=[chatbot_component]
        ).then(lambda: "", None, msg)  # Очищает поле после отправки

    return demo

if __name__ == "__main__":
    demo = create_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860)