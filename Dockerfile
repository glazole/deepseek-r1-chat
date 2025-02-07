# Используем базовый образ Python 3.10
FROM python:3.10

# Устанавливаем PyTorch с поддержкой CUDA
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Создаём рабочую директорию
WORKDIR /app

# Копируем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь код
COPY . /app

# Запускаем Gradio
CMD ["python", "app.py"]