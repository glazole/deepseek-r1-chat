#!/bin/bash
# Проверка доступности GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA-SMI not found. GPU may not be available."
else
    nvidia-smi
fi

# Запуск Ollama в фоне
ollama serve &

# Даем ему время на запуск
sleep 35

# Загружаем модели при старте (если они не загружены)
ollama list | grep "deepseek-r1:1.5b" || ollama pull deepseek-r1:1.5b

# Запуск вашего Python приложения
python3 app.py
