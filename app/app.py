import logging
from demo import create_demo
from ollama_test import test_ollama_connection


# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# Проверяем доступность Ollama перед запуском
if not test_ollama_connection():
    exit(1) 

if __name__ == "__main__":
    demo = create_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860)