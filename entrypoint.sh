#!/bin/sh
# Запускаем Ollama в фоне
ollama serve &
# Даем ему время на запуск
sleep 5
# Загружаем модели при старте (если они не загружены)
ollama list | grep "deepseek-r1:1.5b" || ollama pull deepseek-r1:1.5b
ollama list | grep "deepseek-r1:3b" || ollama pull deepseek-r1:3b
# Запускаем Gradio
exec python app.py
