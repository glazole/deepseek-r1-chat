import gradio as gr
import numpy as np
import requests
import logging
import re
import time
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
from langchain_core.messages import HumanMessage, AIMessage

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# URL Ollama API
OLLAMA_API = "http://ollama:11434"

# Проверка подключения к Ollama перед запуском
def test_ollama_connection(retries=3, delay=3):
    """
    Проверяет доступность сервера Ollama перед запуском приложения.

    Аргументы:
        retries (int): Количество попыток подключения перед отказом (по умолчанию 3).
        delay (int): Время ожидания (в секундах) между повторными попытками (по умолчанию 3 сек).

    Возвращает:
        bool: True, если подключение успешно; False, если не удалось подключиться после всех попыток.
    """
    for i in range(retries):
        try:
            # Отправляем GET-запрос к API Ollama
            response = requests.get(f"{OLLAMA_API}/api/tags", timeout=5)

            # Если ответ успешен (статус 200), значит сервер работает
            response.raise_for_status()
            logging.info("✅ Успешное подключение к Ollama")
            return True  # Подключение успешно

        except requests.exceptions.RequestException as e:
            # Если произошла ошибка (например, сервер не отвечает), логируем предупреждение
            logging.warning(f"❌ Ошибка подключения ({i+1}/{retries}): {e}")

            # Ждем перед повторной попыткой
            time.sleep(delay)

    # Если все попытки не удались, логируем ошибку и возвращаем False
    logging.error("🚨 Не удалось подключиться к Ollama после нескольких попыток")
    return False

# Проверяем доступность Ollama перед запуском
if not test_ollama_connection():
    exit(1) 

def get_llm_engine(model_name, temperature, top_p, top_k, repetition_penalty):
    """
    Инициализирует LLM-модель с заданными параметрами.

    Аргументы:
        model_name (str): Название модели (например, "deepseek-r1:1.5b").
        temperature (float): Температура генерации.
        top_p (float): Ограничение по вероятностному порогу (Nucleus Sampling).
        top_k (int): Количество возможных токенов (Top-K Sampling).
        repetition_penalty (float): Штраф за повторяющиеся слова.

    Возвращает:
        ChatOllama: Объект модели, если инициализация успешна.
        None: Если произошла ошибка.
    """
    try:
        return ChatOllama(
            model=model_name,
            base_url=OLLAMA_API,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,  # Новый параметр
        )
    except Exception as e:
        logging.error(f"❌ Ошибка инициализации модели {model_name}: {e}")
        return None

# Настройки системного промпта
SYSTEM_TEMPLATE = """You are an expert AI coding assistant. Provide concise, correct solutions 
with strategic print statements for debugging. Always respond in English."""

chat_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(SYSTEM_TEMPLATE),
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessagePromptTemplate.from_template("{input}")
])

# Класс чат-бота
class ChatBot:
    """
    Класс ChatBot управляет историей чата и взаимодействием с LLM-моделью.
    Он отправляет сообщения в модель, анализирует ответы и сохраняет их.
    """

    def __init__(self):
        """
        Инициализация чата. История начинается с приветственного сообщения от ассистента.
        """
        self.chat_history = [
            {"role": "assistant", "content": "Hi! I'm **DeepSeek**. How can I help you code today? 💻"}
        ]

    def generate_ai_response(self, user_input, llm_engine):
        """
        Отправляет запрос модели, получает ответ и разбирает размышления (<think>...</think>).

        Аргументы:
            user_input (str): Входное сообщение от пользователя.
            llm_engine (ChatOllama): Инициализированная LLM-модель.

        Возвращает:
            tuple: (размышления модели, основной ответ).
        """
        logging.info(f"📝 Отправка запроса в модель: {user_input}")

        # Добавляем сообщение пользователя в историю в формате HumanMessage
        self.chat_history.append(HumanMessage(content=user_input))

        # Создаем цепочку вызова LangChain: промпт → модель → обработчик строк
        chain = chat_prompt | llm_engine | StrOutputParser()

        # Отправляем запрос и получаем ответ, если ответа нет — возвращаем сообщение об ошибке
        response = chain.invoke({"input": user_input, "chat_history": self.chat_history}) or "⚠️ Ошибка: модель не вернула ответ."

        # Разбираем ответ модели, выделяя размышления <think>...</think>
        thoughts_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
        if thoughts_match:
            thoughts = thoughts_match.group(1).strip()  # Извлекаем текст между <think> и </think>
            main_response = response.replace(thoughts_match.group(0), "").strip()  # Убираем размышления из основного ответа
        else:
            thoughts = None  # Если размышлений нет, оставляем None
            main_response = response.strip()  # Убираем лишние пробелы из основного ответа

        logging.info(f"💡 Размышления:\n{thoughts if thoughts else 'Нет размышлений'}")
        logging.info(f"💡 Основной ответ:\n{main_response}")

        # Если модель вернула размышления, добавляем их в историю чата
        if thoughts:
            self.chat_history.append(AIMessage(content=f"🤔 **Model's Thoughts:**\n> {thoughts}"))

        # Добавляем основной ответ в историю чата
        self.chat_history.append(AIMessage(content=main_response))

        return thoughts, main_response  # Возвращаем размышления и основной ответ

    # Чат в Gradio
    def chat(self, message, model_choice, temperature, top_p, top_k, repetition_penalty, history):
        """
        Обрабатывает сообщение пользователя, отправляет его в LLM-модель и обновляет историю чата.

        Аргументы:
            message (str): Входное сообщение от пользователя.
            model_choice (str): Выбранная модель.
            temperature (float): Температура генерации ответа.
            top_p (float): Вероятностное ограничение (nucleus sampling).
            top_k (int): Количество возможных вариантов токенов.
            repetition_penalty (float): Штраф за повторение слов.
            history (list): История чата.

        Возвращает:
            tuple: ("", обновленная история чата)
        """
        if not message:
            return "", history

        # Получаем LLM с выбранными параметрами
        llm_engine = get_llm_engine(model_choice, temperature, top_p, top_k, repetition_penalty)

        history.append({"role": "user", "content": message})

        thoughts, ai_response = self.generate_ai_response(message, llm_engine)

        if thoughts:
            history.append({"role": "assistant", "content": f"🤔 **Model's Thoughts:**\n> {thoughts}"})

        history.append({"role": "assistant", "content": ai_response})

        logging.info(f"📜 Обновленная история чата:\n{history}")

        return "", history
    
    # Только для тестирования чата (без подключения к модели)
    def chat_test(self, message, model_choice, history):
        """
        Тестовая функция для чата, используется без подключения к модели LLM.

        Аргументы:
            message (str): Входное сообщение от пользователя.
            model_choice (str): Выбранная модель (не используется в этой тестовой функции).
            history (list): История чата в формате [{"role": "user"/"assistant", "content": "текст"}].

        Возвращает:
            tuple: ("", обновленная история чата) — пустая строка очищает поле ввода в Gradio.
        """
        # Добавляем сообщение пользователя в историю чата
        history.append({"role": "user", "content": message})

        # Добавляем фиксированный тестовый ответ модели
        history.append({"role": "assistant", "content": "This is a test response."})

        # Возвращаем пустую строку (для очистки поля ввода в Gradio) и обновленную историю чата
        return "", history

# Создание интерфейса Gradio
def create_demo():
    """
    Создает и настраивает веб-интерфейс Gradio с выбором модели и параметров генерации.
    """
    # Создаем экземпляр класса ChatBot
    chatbot = ChatBot()

    # Создаем интерфейс Gradio с темой Soft и кастомными цветами
    with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", neutral_hue="zinc")) as demo:
        # Добавляем заголовок и описание
        gr.Markdown("# 🧠 DeepSeek Code Companion")  # Основной заголовок
        gr.Markdown("🚀 Your AI Pair Programmer with Debugging Superpowers")  # Подзаголовок
            
        with gr.Row():  # Размещаем элементы в строке
            with gr.Column(scale=4):  # Левая колонка (основная часть интерфейса)
                # Чатбот-компонент
                chatbot_component = gr.Chatbot(
                    value=[{"role": "assistant", "content": "Hi! I'm **DeepSeek**. How can I help you code today? 💻"}],  
                    show_copy_button=True,  # Позволяет копировать сообщения
                    height=500,  # Высота чата
                    type="messages",  # Используем формат OpenAI (role + content)
                    render_markdown=True,  # Позволяет рендерить Markdown
                )

                # Поле ввода сообщений от пользователя
                msg = gr.Textbox(placeholder="Type your coding question here...", show_label=False)

                # Кнопка очистки чата (очищает поле ввода и чатбот-компонент)
                clear = gr.ClearButton([msg, chatbot_component])

            with gr.Column(scale=1):  # Правая колонка (выбор модели и описание)
                # Выпадающий список для выбора модели LLM
                model_dropdown = gr.Dropdown(
                    choices=["deepseek-r1:1.5b", "deepseek-r1:7b", "deepseek-r1:14b"],  # Доступные модели
                    value="deepseek-r1:1.5b",  # Модель по умолчанию
                    label="Choose Model"  # Название поля выбора
                )

                # Новый выпадающий список для выбора температуры
                temperature_dropdown = gr.Dropdown(
                    choices=[round(x, 1) for x in np.arange(0.1, 1.0, 0.1)],  # Генерируем шаги 0.1-0.7
                    value=0.3,  # Значение по умолчанию
                    label="Temperature",
                )
                # Новый слайдер для выбора top_p (Nucleus Sampling)
                top_p_slider = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    step=0.1,
                    value=1.0,
                    label="Top-P Sampling"
                )

                # Новый слайдер для выбора top_k (Top-K Sampling)
                top_k_slider = gr.Slider(
                    minimum=1,
                    maximum=100,
                    step=1,
                    value=50,
                    label="Top-K Sampling"
                )

                # Новый слайдер для выбора repetition_penalty
                repetition_penalty_slider = gr.Slider(
                    minimum=1.0,
                    maximum=2.0,
                    step=0.1,
                    value=1.0,
                    label="Repetition Penalty"
                )

                # Описание возможностей модели
                gr.Markdown("### Model Capabilities")  
                gr.Markdown("""
                - 🐍 **Python Expert**
                - 🐞 **Debugging Assistant**
                - 📝 **Code Documentation**
                - 💡 **Solution Design**
                """)

                # Ссылки на используемые технологии
                gr.Markdown("Built with [Ollama](https://ollama.ai/) | [LangChain](https://python.langchain.com/)")

        # Привязываем отправку сообщения к функции chat() у ChatBot
        msg.submit(chatbot.chat, [msg, model_dropdown, temperature_dropdown, top_p_slider, top_k_slider, repetition_penalty_slider, chatbot_component], [msg, chatbot_component])

    return demo  # Возвращаем объект интерфейса

if __name__ == "__main__":
    demo = create_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860)