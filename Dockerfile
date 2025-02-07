# Используйте базовый образ с CUDA 12.4
FROM nvidia/cuda:12.4.0-base-ubuntu22.04

# Установите Python и pip
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*
    
# Устанавливаем Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Установите PyTorch с поддержкой CUDA 12.4
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Скопируйте текущую директорию в /app
COPY . /app
# Перейдите в директорию /app
WORKDIR /app

# Установите другие зависимости Python
COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY entrypoint.sh /entrypoint.sh
# Делаем скрипт исполняемым
RUN chmod +x /entrypoint.sh
# Запустите ваше приложение
CMD ["/entrypoint.sh"]
