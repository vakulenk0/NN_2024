from random import random
from math import sqrt


class Neuron:
    def __init__(self):
        self.nu = 0.5
        self.w = [random(), random()]  # Получаем веса
        divider = sqrt(self.w[0] ** 2 + self.w[1] ** 2)  # Нормируем веса
        self.w[0] /= divider
        self.w[1] /= divider

    def calculate(self, x: list):
        return self.w[0] * x[0] + self.w[1] * x[1]

    def recalculate(self, x: list):
        self.w[0] += self.nu * (x[0] - self.w[0])
        self.w[1] += self.nu * (x[1] - self.w[1])

    def activation_func(self, u):
        pass
        

class NeuralNetwork:
    def __init__(self):
        self.x = [   # Массив входных значений
            [0.97, 0.2],
            [1, 0],
            [-0.72, 0.7],
            [-0.67, 0.74],
            [-0.8, 0.6],
            [0, -1],
            [0.2, -0.97],
            [-0.3, -0.95]
        ]

        self.neurons = [Neuron() for i in range(4)]

    def __str__(self):
        s = ''
        for neuron in self.neurons:
            s += str(neuron.w) + '\n'
        return s

    def start(self, threshold_number):
        u = [0 for i in range(4)]
        number_wins = [0, 0, 0, 0]

        for i in range(len(self.x)):
            for j in range(4):
                if number_wins[j] < threshold_number:
                    u[j] = self.neurons[j].calculate(self.x[i])
                else:
                    u[j] -= 100
            j = u.index(max(u))
            self.neurons[j].recalculate(self.x[i])
            number_wins[j] += 1
            print(j + 1)


nn = NeuralNetwork()
print(nn)
nn.start(3)
print(nn)

