#Карпеченко Д.М. дз09, 2022/10/16 v.0.1
import cv2
from PIL import Image, ImageFilter
import os

#Список наших картинок, с которыми будут происходить изменения (путь к папке с картинками)
img_set = os.listdir(r'./car/')

#Прогоняем по всем картинкам из нашего листа
for i in img_set:
#Переменная с для имени)
    c=0
#Загружаем картинку для работы
    img = Image.open(r'./car/'+i)
    img.load()
#Используем фильтр размытия
    img1 = img.filter(ImageFilter.BLUR)
    c+=1

#Сохраняем картинку в папку под именем 2)
    img1.save(r'./2/'+str(c)+i)
#Делаем из цветной картинки черно-белую
    img1 = img.convert('L')
    c+=1
    img1.save(r'./2/'+str(c)+i)
    #img1.show()
#Делаем из картинки красивую гравюрную штуку
    img1 = img.filter(ImageFilter.FIND_EDGES)
    c+=1
    img1.save(r'./2/'+str(c)+i)
    c+=1
#Меняем масштаб картинки
    img1 = img.reduce(3)
    img1.save(r'./2/' + str(c) + i)
    c=0

