import numpy as np
import matplotlib.pyplot as plt
# region  кусок с вычислениями

x = np.arange(-3, 3, 0.1) # диапазон x
y = np.arange(-3, 3, 0.1) # диапазон y

X, Y = np.meshgrid(x, y) # координатные оси
Z = X**2 - Y**2 + 2*Y # исходные данные

r = np.random.normal(0, 2, size=Z.shape) # белый шум
F = Z + r # данные + шум
# f для того, чтобы в один график добавить и исходные, и с шумом данные

A = np.ones((len(x) * len(y), 4)) # массив
A[:, 1] = X.flatten()**2 # сглаживание?
A[:, 3] = Y.flatten()**2
A[:, 2] = Y.flatten() 

result = np.linalg.lstsq(A, F.flatten()) # массив результатов

ZA = np.dot (A, result[0]).reshape(F.shape) # аппроксимирующая поверхность

# endregion

# region все графики на одной картинке

fig = plt.figure () # поле для графика
ax = fig.add_subplot (111, projection='3d') 
ax.plot_surface (X, Y, Z, cmap = 'inferno', alpha = 0.5, label ='Исходные данные') # исходный график
ax.plot_surface (X, Y, F, cmap = 'viridis', alpha = 0.5, label = 'Исходные данные + ШУМ') # график с шумом
ax.plot_surface (X, Y, ZA, color = 'blue', alpha = 0.5, label = 'Аппроксимация') # график апроксимации
ax.set_xlabel ('x') # подписать оси
ax.set_ylabel ('y')
ax.set_zlabel ('z')
ax.legend () # добавить легенду 
plt.savefig ('Все_данные.png') # сохранение картинки
plt.show () # вывод на экран
#endregion

# region график шум + апроксимация 

fig = plt.figure ()
ax = fig.add_subplot (111, projection = '3d')
ax.plot_surface (X, Y, F, cmap = 'viridis', alpha = 0.5, label = 'Исходные данные + ШУМ') # график с шумом
plt.savefig ('Исходные_данные+ШУМ.png')
ax.plot_surface (X, Y, ZA, color = 'blue', alpha = 0.5, label = 'Аппроксимация')# график апроксимации
plt.savefig ('Аппроксимация.png')
ax.set_xlabel ('x')
ax.set_ylabel ('y')
ax.set_zlabel ('z')
ax.legend ()
plt.show ()
#endregion

# region графики отдельно    

fig = plt.figure()

ax1 = fig.add_subplot (131, projection = '3d') # исходный график
ax1.plot_surface (X, Y, Z, cmap = 'inferno', alpha = 0.5, label = 'Исходные данные')
ax1.set_xlabel ('x')
ax1.set_ylabel ('y')
ax1.set_zlabel ('z')
ax1.legend ()

ax2 = fig.add_subplot (132, projection='3d') # график с шумом
ax2.plot_surface (X, Y, F, cmap = 'viridis', alpha = 0.5, label = 'Исходные данные + ШУМ')
ax2.set_xlabel ('x')
ax2.set_ylabel ('y')
ax2.set_zlabel ('z')
ax2.legend ()

ax3 = fig.add_subplot (133, projection = '3d') # график апроксимации
ax3.plot_surface (X, Y, ZA, color ='blue', alpha = 0.5, label = 'Аппроксимация')
ax3.set_xlabel ('x')
ax3.set_ylabel ('y')
ax3.set_zlabel ('z')
ax3.legend ()

plt.savefig ('ОТДЕЛЬНЫЕ ГРАФИКИ.png')# отображение графиков
plt.show ()
#endregion

#region вывод результатов в терминал

# print ('                           ', '|свободный член|', '|коэфф при х|', '|коэфф при у|')
print ('Коэффициенты аппроксимации:', result[0]) #коэфф апроксимации
print ('Ошибка апроксимации:', result[1] / (len(x) * len(y))) # ошибка аппроксимации
print ('Кортеж значений:', result)
#endregion