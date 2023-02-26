#!/usr/bin/env python
# coding: utf-8

# ДЗ_27 Бенатович Е.Ю.
# 
# 
# Обучение нейронной сети на наборе данных notMNIST_small

# In[1]:


import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D, MaxPooling2D
from keras.models import Sequential
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer


# In[2]:


# путь к папкам с изображениями
PATH = './notMNIST_small'

classes = os.listdir(PATH)
num_classes = len(classes)

print("Всего {} классов: {}".format(num_classes, classes))


# In[3]:


X = [] # изображения
y = [] # метки
for directory in classes:
    for image in os.listdir(PATH + '/' + directory):
        try:
            path = PATH + '/' + directory + '/' + image
            img = Image.open(path)
            img.load()
            img_X = np.asarray(img, dtype=np.int16)
            X.append(img_X)
            y.append(directory)
        except:
            None


# In[4]:


X = np.asarray(X)/255  # выполним нормализацию значений
y = np.asarray(y)


# In[5]:


os.listdir(PATH)


# In[6]:


X.shape, y.shape


# In[7]:


num_images = len(X)
size = len(X[0])


# In[8]:


# выведем 10 случайных изображений
for j in range(10):
    i=random.randint(0,num_images)
    plt.subplot (5,5,j+1)
    plt.subplots_adjust(wspace=0.3, hspace=1.2)
    plt.imshow(X[i], cmap='gray')
    plt.title("Буква {}".format(y[i]));
    


# In[9]:


y


# In[10]:


# конвертируем метки из целых чисел в векторы
lb = LabelBinarizer()
y= lb.fit_transform(y)


# In[11]:


y


# In[12]:


y.shape


# In[38]:


# разобьем датасет на тренировочную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=102)


# In[39]:


physical_devices = tf.config.list_physical_devices('GPU')


# In[40]:


tf.keras.backend.clear_session()
np.random.seed(102)
tf.random.set_seed(102)


# In[41]:


callbacks = tf.keras.callbacks.EarlyStopping(
    monitor='accuracy',
    min_delta=0.01,
    patience=30,
    verbose=0,
    mode='auto',
    baseline=None,
    restore_best_weights=False
)


# #### Model - Multilayer Perceptron

# In[45]:


model_mlp = keras.Sequential([
    Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2), strides=2),
    Conv2D(64, (3,3), padding='same', activation='relu'),
    MaxPooling2D((2, 2), strides=2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10,  activation='softmax')
])

model_mlp.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005), metrics=['accuracy'], loss='categorical_crossentropy')
model_mlp.summary()


# In[46]:


model_mlp.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=1000, callbacks=callbacks)


# In[47]:


model_mlp.evaluate(X_test, y_test, verbose=0)


# #### Model - Convolutional Neural Network

# In[59]:


model_cnn = Sequential([
    Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)),
    MaxPool2D(),
    Conv2D(filters=64, kernel_size=(3,3), activation='relu'),
    MaxPool2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(10, activation='softmax'),
])

model_cnn.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'], loss='categorical_crossentropy')
model_mlp.summary()


# In[60]:


history=model_cnn.fit(X_train, y_train, epochs=20, validation_split=0.3, batch_size=256, callbacks=callbacks, verbose=1)


# In[61]:


model_cnn.save('cnn_model')


# In[62]:


model_cnn.evaluate(X_test, y_test, verbose=0)


# In[63]:


print("Accuracy {} %".format(model_cnn.evaluate(X_test, y_test, verbose=0)[1]*100))


# In[64]:


history.history.keys()


# In[65]:


plt.plot(history.history['accuracy'], label='Точность обучающая выборка')
plt.plot(history.history['val_accuracy'], label='Точность тестовая выборка')
plt.xlabel('Epoch')
plt.ylabel('accuracy')
plt.legend()
plt.show()


# In[66]:


plt.plot(history.history['loss'], label='Потери обучающая выборка')
plt.plot(history.history['val_loss'], label='Потери тестовая выборка')
plt.xlabel('Epoch')
plt.ylabel('accuracy')
plt.legend()
plt.show()


# Model - Multilayer Perceptron 91.42%
# 
# Model - Convolutional Neural Network 92.13%
# 

# In[ ]:




