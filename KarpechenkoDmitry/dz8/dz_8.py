#!/usr/bin/env python
# coding: utf-8

# Карпеченко Д.М., dz08, v.0.1, 12.10.2022

# In[66]:


import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
#Снять ограничения на вывод колонок
pd.set_option('display.max_columns', None)
#Снять ограничения на вывод строк
pd.set_option('display.max_rows', 8)
data = pd.read_excel('D:\TeachMeSkills\DZ\dz8\data1.xlsx')
#Удаляем из набора данных столбец порядкового номера записи
filt_data = data.drop(['No'], axis=1)
###print(filt_data.describe())
#Выведем количество не нулевых и не NAN начений в столбцах
print(filt_data.replace('', np.nan).count())

#Проверяем наши данные на критерий 3 сигм. Хотя для данного дата сета наверное нет особого смысла убирать выбрасы,
#т.к. тут они могут нести с собой полезную информацию.

filt_data=filt_data[(np.abs(stats.zscore(filt_data))<3).all(axis=1)]
filt_data.dropna(inplace=True)

#PS нулевых и нан значений нет



# In[3]:


filt_data.describe()


# # Построим графики распределений наших данных

# In[24]:


###print(len(filt_data['X2 house age'].value_counts()))
ax = filt_data['X2 house age'].plot.hist(bins=10, alpha=0.8)


# In[28]:


print(len(filt_data['X3 distance to the nearest MRT station'].value_counts()))
ax = filt_data['X3 distance to the nearest MRT station'].plot.hist(bins=20, alpha=0.8)


# Все хотят поближе к метро, но непонятный оживляш на 2000 и 4000. 

# In[35]:


print(len(filt_data['X4 number of convenience stores'].value_counts()))
ax = filt_data['X4 number of convenience stores'].plot.hist(bins=10, alpha=0.8)


# Количество магазинов в пешей доступности особо не влияет на выбор квартир

# In[26]:


print(len(filt_data['X5 latitude'].value_counts()))
ax = filt_data['X5 latitude'].plot.hist(bins=10, alpha=0.8)


# In[17]:


print(len(filt_data['X6 longitude'].value_counts()))
ax = filt_data['X6 longitude'].plot.hist(bins=50, alpha=0.8)


# Можно предположить, что все хотят урвать квартиру ближе к центру

# In[27]:


print(len(filt_data['Y house price of unit area'].value_counts()))
ax = filt_data['Y house price of unit area'].plot.hist(bins=10, alpha=0.8)


# Распределение по стоимости похоже на нормальное

# # Пронормируем наши значения используя StandardScaler(), для сохранения формы нашего распределения. Но для дальнейших пунктов буду использовать не нормированные данные (для большего моего понимания:))

# In[37]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
filt1_data = scaler.fit_transform(filt_data)


# In[46]:


sns.distplot(filt1_data[:,-1], bins = 17);


# # Построим таблицу корелляции

# In[59]:


corr_matrix = filt_data.corr()
corr_matrix


# In[57]:


sns.heatmap(corr_matrix, annot=True, linewidths=1);


# # Построим графики парных отношений 

# In[50]:


sns.set(style='white', font_scale=1.3)

g = sns.PairGrid(filt_data, diag_sharey=False)
g.map_lower(sns.kdeplot, cmap='Blues_d')
g.map_upper(plt.scatter, alpha=0.9)
g.map_diag(sns.kdeplot, lw=3);


# # Проведем тесты на нормальность по Пирсону

# In[68]:


stat, p = stats.normaltest(filt_data['X2 house age'])
print('Statistics=%.3f, p-value=%.3f' % (stat, p))

alpha = 0.05
if p > alpha:
    print('Принять гипотезу о нормальности')
else:
    print('Отклонить гипотезу о нормальности')


# In[65]:


stat, p = stats.normaltest(filt_data['Y house price of unit area'])
print('Statistics=%.3f, p-value=%.3f' % (stat, p))

alpha = 0.05
if p > alpha:
    print('Принять гипотезу о нормальности')
else:
    print('Отклонить гипотезу о нормальности')


# На нормальность прошел тест для стоимости. Для всех остальных данных тест не прошел (сюда не включил, чтобы не загромождать)

# # Вывод

# Из полученных данных можно заметить, что между собой кореллируют данные по расположению к метро, координатам, количеством магазинов в пешей доступности и стоимости. Получили отрицательную корелляцию (чем ближе к метро, тем стоимость квартир выше, а также больше магазинов :)) Есть также слабая зависимость между стоимостью жилья и количеством магазинов в округе. Также можно предположить, что в центре Нового Тайбэя количество станций больше и кучнее, чем на окраине)
# Тест на нормальность (по Пирсону) прошли только данные о стоимости жилья.
