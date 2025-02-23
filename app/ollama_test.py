import logging
import os
import time
import requests


# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# URL Ollama API
OLLAMA_API = os.getenv('OLLAMA_API')

# Проверка подключения к Ollama перед запуском
def test_ollama_connection(retries=3, delay=3, url: str = OLLAMA_API):
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
            response = requests.get(f"{url}/api/tags", timeout=5)

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