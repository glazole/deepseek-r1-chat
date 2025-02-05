#!/bin/sh
# Запускаем Ollama в фоне
ollama serve &
# Даем ему время на запуск
sleep 5
# Запускаем Gradio
exec python app.py
