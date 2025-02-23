import logging

import gradio as gr
import numpy as np
from chat_manager import ChatBot


# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

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

                # Выпадающий список для выбора температуры
                temperature_dropdown = gr.Dropdown(
                    choices=[round(x, 1) for x in np.arange(0.1, 1.0, 0.1)],  # Генерируем шаги 0.1-0.7
                    value=0.3,  # Значение по умолчанию
                    label="Temperature",
                )
                # Слайдер для выбора top_p (Nucleus Sampling)
                top_p_slider = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    step=0.1,
                    value=1.0,
                    label="Top-P Sampling"
                )

                # Слайдер для выбора top_k (Top-K Sampling)
                top_k_slider = gr.Slider(
                    minimum=1,
                    maximum=100,
                    step=1,
                    value=50,
                    label="Top-K Sampling"
                )

                # Слайдер для выбора repeat_penalty
                repeat_penalty_slider = gr.Slider(
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
        msg.submit(chatbot.chat,
                   [msg, model_dropdown, temperature_dropdown, top_p_slider, top_k_slider, repeat_penalty_slider,
                    chatbot_component], [msg, chatbot_component])

    return demo  # Возвращаем объект интерфейса