#number 1
import numpy as np
def slau(A,B):
    print(np.linalg.inv(A))
    A_Inv = np.linalg.inv(A)
    # x=A^-1*B
    print()
    print(A_Inv.dot(B))
    X = A_Inv.dot(B)
    print()
    print(A.dot(X))


A = np.arange(9).reshape(3, 3)
for i in range(3):
   for j in range(3):
     number = int(input("Please Enter Elements of Matrix A:"))
     A[i][j] = number
print (A)
print()
B = np.arange(3).reshape(3, 1)
for i in range(3):
   for j in range(1):
     number=int(input("Please Enter Elements of Vector B:"))
     B[i][j]=number
print()
print(B)
slau(A,B)

#number 2
import numpy as np
import csv
#matrix = np.random.sample((5, 5))
#np.savetxt('matrix.csv', np.linalg.inv(matrix))
#matrix = np.loadtxt('input.csv')
#print(matrix)

# read csv
mat = np.genfromtxt("input.csv", delimiter=";", encoding="UTF-8")
# display the array
print(mat)
mat_inv = np.linalg.inv(mat)
print(mat_inv)
np.savetxt('output.csv', mat_inv, delimiter=';')
#number 3
print()

# Import libraries
import matplotlib.pyplot as plt
import numpy as np

# Creating vectors X and Y
x = np.linspace(-20, 20, 50)
fig = plt.figure(figsize=(14, 8))

y = np.cos(x)
# Create the plot
plt.plot(x, y, 'b', label = 'cos(x)')

y2 = 1-x**2/2
plt.plot(x, y2, 'r-.', label = 'Degree 2')

y4 = 1-x**2/2 + x**4 / 24
plt.plot(x, y4, 'g:', label = 'Degree 4')

y6 = 70 + 11*x - 10*x**2 + x**3
plt.plot(x,y6)#sample

coeff = [ 1. , -10. , 11., 70. ]
print('Корни полинома:', np.roots(coeff))


plt.legend()
plt.grid(True, linestyle = ':')
plt.xlim([-6, 6])
plt.ylim([-4,4])

plt.title('Taylor Polinomias of cos(x) at x = 0')
plt.xlabel('x-axis')
plt.ylabel('y-axis')

# Show the plot
plt.show()

print()

import numpy as np
import matplotlib.pyplot as plt

theta = np.radians(np.linspace(0,360*5,1000))
r = theta**2
x_2 = r*np.cos(theta)
y_2 = r*np.sin(theta)
plt.figure(figsize=[10,10])
plt.plot(x_2,y_2)
plt.show()


print()#полярныее координаты

import numpy as np
import matplotlib.pyplot as plt

plt.axes(projection='polar')

# setting the length
# and number of petals
a = 1
n = 6

# creating an array
# containing the radian values
rads = np.arange(0, 2 * np.pi, 0.001)

# plotting the rose
for rad in rads:
    r = a * np.cos(n * rad)
    plt.polar(rad, r, 'g.')

# display the polar plot
plt.show()

import numpy as np

#number 4

print()

#point 1.1
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.stats import multivariate_normal

X = np.linspace(-5,5,500)
Y = np.linspace(-5,5,500)
X, Y = np.meshgrid(X,Y)
X_mean = 0; Y_mean = 0
X_var = 5; Y_var = 8
pos = np.empty(X.shape+(2,))
pos[:,:,0]=X
pos[:,:,1]=Y
rv = multivariate_normal([X_mean, Y_mean],[[X_var, 0], [0, Y_var]])
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, rv.pdf(pos), cmap="plasma")

plt.show()

#point 1.2
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy

def makeData ():
    # Строим сетку в интервале от -10 до 10, имеющую 100 отсчетов по обоим координатам
    x = numpy.linspace (-10, 10, 100)
    y = numpy.linspace (-10, 10, 100)

    # Создаем двумерную матрицу-сетку
    xgrid, ygrid = numpy.meshgrid(x, y)

    z = numpy.sin (xgrid) * numpy.sin (ygrid) / (xgrid * ygrid)
    return xgrid, ygrid, z
if __name__ == '__main__':
    x, y, z = makeData()
    fig = plt.figure()
    axes = fig.add_subplot(projection='3d')
    cmap = LinearSegmentedColormap.from_list ('red_blue', ['black', 'white', 'pink'], 256)
    axes.plot_surface(x, y, z, color='#11aa55', cmap=cmap)

    plt.show()

    #point 2
    print()

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for c, z in zip(['black', 'white', 'pink', 'blue'], [30, 20, 10, 0]):
    xs = np.arange(20)
    ys = np.random.rand(20)

    # You can provide either a single color or an array. To demonstrate this,
    # the first bar of each set will be colored cyan.
    cs = [c] * len(xs)
    cs[0] = 'c'
    ax.bar(xs, ys, zs=z, zdir='y', color=cs, alpha=0.8)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()

#point 3

print()

import numpy as np
import matplotlib.pyplot as plt


ax = plt.figure().add_subplot(projection='3d')

# Prepare arrays x, y, z
theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
z = np.linspace(-2, 2, 100)
r = z**2 + 1
x = r * np.sin(theta)
y = r * np.cos(theta)

ax.plot(x, y, z, label='parametric curve')
ax.legend()

plt.show()

#number 5
print()

#point 1

from sympy import *
import numpy as np
import matplotlib.pyplot as plt

x = Symbol('x')
y = x**2 + 1

yprime = y.diff(x)
print("Результат вычислений производной: ", yprime)

res = integrate(2*x, x)
print("Результат вычислений интеграла: ", res)

#point 2
import matplotlib.pyplot as plt

tex = '$\\frac{1}{\\sqrt{2\\sqrt{2\\pi}}} \\exp\\left(-\\frac{(x-\\mu)^2}{2\\sigma^2}\\right)$'

### Создание области отрисовки
fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
ax.set_axis_off()

### Отрисовка формулы
t = ax.text(0.5, 0.5, tex,
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=20, color='black')

### Определение размеров формулы
ax.figure.canvas.draw()
bbox = t.get_window_extent()
print (bbox.width, bbox.height)

# Установка размеров области отрисовки
fig.set_size_inches(bbox.width / 80, bbox.height / 80)  # dpi=80

### Отрисовка или сохранение формулы в файл
plt.show()


#защита
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

#theta = np.radians(np.linspace(0,360*5,1000))
theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
z = np.linspace(-2, 2, 100)
r = z**2 + 1
x_2 = r*np.sin(theta)
y_2 = r*np.cos(theta)

cubic_interpolation_model = interp1d(x_2, y_2, kind = "cubic")

plt.figure(figsize=[10,10])
plt.plot(x_2,y_2)



#x = r*np.sin(theta)
#y = r*np.cos(theta)
#plt.plot(x,y)

plt.show()


