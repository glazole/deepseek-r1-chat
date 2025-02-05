# Используем базовый образ Python 3.10
FROM python:3.10

# Устанавливаем зависимости для системы
RUN apt-get update && apt-get install -y curl wget

# Устанавливаем Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Создаем рабочую директорию
WORKDIR /app

# Копируем файлы проекта
COPY . .

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Загружаем модель DeepSeek-r1
RUN ollama pull deepseek-r1:1.5b
RUN ollama pull deepseek-r1:3b

# Открываем порт для Gradio
EXPOSE 7860

# Копируем entrypoint.sh в контейнер
COPY entrypoint.sh /entrypoint.sh
# Делаем скрипт исполняемым
RUN chmod +x /entrypoint.sh
# Запускаем Ollama и приложение
CMD ["/entrypoint.sh"]
