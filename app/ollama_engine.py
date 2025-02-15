import logging
import os
from langchain_ollama import ChatOllama


# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# URL Ollama API
OLLAMA_API = os.getenv('OLLAMA_API')

def get_llm_engine(model_name: str, temperature: float, top_p: float, top_k: int, repeat_penalty: float, url: str =  OLLAMA_API):
    """
    Инициализирует LLM-модель с заданными параметрами.

    Аргументы:
        model_name (str): Название модели (например, "deepseek-r1:1.5b").
        temperature (float): Температура генерации.
        top_p (float): Ограничение по вероятностному порогу (Nucleus Sampling).
        top_k (int): Количество возможных токенов (Top-K Sampling).
        repeat_penalty (float): Штраф за повторяющиеся слова.

    Возвращает:
        ChatOllama: Объект модели, если инициализация успешна.
        None: Если произошла ошибка.
    """
    try:
        return ChatOllama(
            model=model_name,
            base_url=url,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
        )
    except Exception as e:
        logging.error(f"❌ Ошибка инициализации модели {model_name}: {e}")
        return None