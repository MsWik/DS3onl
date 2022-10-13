## Группа DS03-onl

__Студент Парфимович Алексей__

## Домашнее задание №8

Скачать данные из репозитория UCI (для задач регрессии) и подготовить данные для модели:
1. Нормализовать данные (Скалировать данные в диаппазоне от -1 до 1).
2. Обработать пропуски и выбросы.
3. Построить матрицу корреляции.
4. Построить гистограммы распределения.
5. Выполнить поиск взаимозависимых данных.
6. Выполнить тест на нормальность.
7. Написать вывод по полученным результатам


## Real estate valuation data set Data Set
### Abstract: 
The real estate valuation is a regression problem. 
The market historical data set of real estate valuation are collected from Sindian Dist., New Taipei City, Taiwan.

- Data Set Characteristics: Multivariate
- Number of Instances: 414
- Area: Business
- Attribute Characteristics: Integer, Real
- Number of Attributes: 7
- Date Donated 2018-08-18
- Associated Tasks: Regression
- Missing Values? N/A
- Number of Web Hits: 161988

### Attribute Information:
#### The inputs are as follows
- X1=the transaction date (for example, 2013.250=2013 March, 2013.500=2013 June, etc.)
- X2=the house age (unit: year)
- X3=the distance to the nearest MRT station (unit: meter) - расстояние до станции метро
- X4=the number of convenience stores in the living circle on foot (integer) - магазины в шаговой доступности
- X5=the geographic coordinate, latitude. (unit: degree)
- X6=the geographic coordinate, longitude. (unit: degree)
#### The output is as follow
- Y= house price of unit area (10000 New Taiwan Dollar/Ping, where Ping is a local unit, 1 Ping = 3.3 meter squared) стоимость за 1 пинг (3,3 м2)