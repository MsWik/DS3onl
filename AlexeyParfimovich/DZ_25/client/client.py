#!/usr/bin/env python3

# Импортируем системную библиотеку Python для загрузки файла 'index.html' с сервера

import urllib.request

# Переменная fp содержит запрос к 'http://localhost:1234/'

fp = urllib.request.urlopen("http://localhost:1234/")

# Закодированный ответ сервера ('index.html')
encodedContent = fp.read()

# Раскодированный ответ сервера
decodedContent = encodedContent.decode("utf8")

# Выводим содержимое файла, полученного с сервера ('index.html')
print(decodedContent)

# Закрываем соединение с сервером
fp.close()