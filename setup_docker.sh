#!/bin/bash

# Скрипт для подготовки папок для Docker развертывания

echo "Создание папок для Docker..."

# Создаем папки если их нет
mkdir -p checkpoints
mkdir -p .dvc
mkdir -p .git

# Устанавливаем правильные права
chmod 755 checkpoints
chmod 755 .dvc
chmod 755 .git

echo "Папки созданы успешно!"
echo "Теперь можно запускать docker-compose up --build" 