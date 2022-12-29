import pandas as pd

# подключаем тренировочный датасет
df = pd.read_csv('train.csv')

# анализируем датасет
print(df.head())
print(df.info())
print(df.isna().sum())

# изучим целевой признак (узнаем сколько категорий в целевом столбике)
print(df['price_range'].value_counts())

# выделяем целевой признак
X = df.drop('price_range', axis=1)
y = df['price_range']

# разделяем данные на тренировочные и тестовые
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# тренируем модель МО по алгоритму KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 17)
knn.fit(X_train,y_train)

# получаем предсказания по тестовым данным
y_pred = knn.score(X_test,y_test)
print('Точность предсказаний по методу ближайших соседей: ', y_pred)

# найдем оптимальное значение n ближайших соседей
import numpy as np
import matplotlib.pyplot as plt
# error_rate = []
# for i in range(1,20):
    
#     knn = KNeighborsClassifier(n_neighbors=i)
#     knn.fit(X_train,y_train)
#     pred_i = knn.predict(X_test)
#     error_rate.append(np.mean(pred_i != y_test))

# # визуализируем показатель ошибки
# plt.figure(figsize=(10,6))
# plt.plot(range(1,20),error_rate,color='blue', linestyle='dashed', marker='o',
#          markerfacecolor='red', markersize=5)
# plt.title('Зависимость ошибки от числа ближайших соседей')
# plt.xlabel('Число ближайшие соседи')
# plt.ylabel('Значение ошибки')
# plt.show()

# # создадим новый датафрейм с предсказанными и известными результатами
# y_pred2 = knn.predict(X_test)
# df2 = pd.DataFrame({'known':y_test, 'predidcted':y_pred2})
# print(df2.head(20))

# # визуализируем результаты и ошибки 
# df2.value_counts().plot(kind = 'pie', autopct='%1.1f%%')
# plt.show()

# опорные вектора
from sklearn import svm
clf = svm.SVC()
clf.fit(X_train,y_train)
y_pred2 = clf.predict(X_test)
y_pred = clf.score(X_test, y_test)
print('Точность предсказаний по алгоритму Опорных Векторов: ', y_pred)

# алгоритм логистической регрессии
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train,y_train)
y_pred = lr.score(X_test, y_test)
print('Точность предсказаний по алгоритму Логистической регрессии: ', y_pred)

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)
y_pred = rfc.score(X_test,y_test)
print('Точность предсказаний по алгоритму Случайный ЛЕС: ', y_pred)

# зависимость стоимости от памяти
import seaborn as sns
sns.pointplot(y="int_memory", x="price_range", data=df)
plt.show()

# зависимость стоимости от батарейки
sns.boxplot(x="price_range", y="battery_power", data=df)
plt.show()