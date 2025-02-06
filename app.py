import gradio as gr
from langchain_ollama import ChatOllama
import requests  # Добавил import

# Тест Ollama перед запуском Gradio
def test_ollama_connection():
    url = "http://127.0.0.1:11434/api/chat"
    payload = {
        "model": "deepseek-r1:1.5b",
        "messages": [{"role": "system", "content": "You are an expert Python and ML/AI coding assistant"}, {"role": "user", "content": "Hello!"}]
    }
    try:
        response = requests.post(url, json=payload)
        print("Test response:", response.json())  # Выводит в лог ответ от модели
    except Exception as e:
        print("Error connecting to Ollama:", e)

# Вызов тестовой функции
test_ollama_connection()

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Initialize the chat engine
def get_llm_engine(model_name):
    return ChatOllama(
        model=model_name,
        base_url="http://127.0.0.1:11434/api/chat",
        temperature=0.3
    )

# System prompt configuration
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
        # Add user message to chat history
        self.chat_history.append(HumanMessage(content=user_input))
        
        # Generate response
        chain = chat_prompt | llm_engine | StrOutputParser()
        response = chain.invoke({
            "input": user_input,
            "chat_history": self.chat_history
        })
        
        # Add AI response to chat history
        self.chat_history.append(AIMessage(content=response))
        return response

    def chat(self, message, model_choice, history):
        if not message:
            return "", history
        
        print(f"[DEBUG] User input: {message}")
        print(f"[DEBUG] Selected model: {model_choice}")

        llm_engine = get_llm_engine(model_choice)
        print("[DEBUG] LLM engine initialized")
        
        # Add user message to log
        self.message_log.append({"role": "user", "content": message})
        
        # Generate AI response
        ai_response = self.generate_ai_response(message, llm_engine)
        print(f"[DEBUG] AI response: {ai_response}")
        
        # Add AI response to log
        self.message_log.append({"role": "ai", "content": ai_response})
        
        # Update chat history
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
                    height=500
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