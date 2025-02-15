import logging
import re
from ollama_prompt import chat_prompt
from ollama_engine import get_llm_engine
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage


# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


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
        response = chain.invoke(
            {"input": user_input, "chat_history": self.chat_history}) or "⚠️ Ошибка: модель не вернула ответ."

        # Разбираем ответ модели, выделяя размышления <think>...</think>
        thoughts_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
        if thoughts_match:
            thoughts = thoughts_match.group(1).strip()  # Извлекаем текст между <think> и </think>
            main_response = response.replace(thoughts_match.group(0),
                                             "").strip()  # Убираем размышления из основного ответа
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
    def chat(self, message, model_choice, temperature, top_p, top_k, repeat_penalty, history):
        """
        Обрабатывает сообщение пользователя, отправляет его в LLM-модель и обновляет историю чата.

        Аргументы:
            message (str): Входное сообщение от пользователя.
            model_choice (str): Выбранная модель.
            temperature (float): Температура генерации ответа.
            top_p (float): Вероятностное ограничение (nucleus sampling).
            top_k (int): Количество возможных вариантов токенов.
            repeat_penalty (float): Штраф за повторение слов.
            history (list): История чата.

        Возвращает:
            tuple: ("", обновленная история чата)
        """
        if not message:
            return "", history

        # Получаем LLM с выбранными параметрами
        llm_engine = get_llm_engine(model_choice, temperature, top_p, top_k, repeat_penalty)

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
