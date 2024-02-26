import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Производная функции активации
def sigmoid_derivative(x):
    return x * (1 - x)

# Входные данные
inputs = np.array(
    [[1, 0, 1],
     [0, 1, 0],
     [0, 1, 0]]
)

# Выходные данные
outputs = np.array(
    [[0],
     [0],
     [1]]
)

# Инициализация весов
weights = np.random.random((3, 1))
print("weights:\n " + str(weights))
# Обучение нейронной сети
for i in range(10):
    # Прямое распространение
    hidden_input = np.dot(inputs, weights)
    print("hidden_input:\n " + str(hidden_input))
    hidden_output = sigmoid(hidden_input)

    # Ошибка
    error = outputs - hidden_output

    # Корректировка весов
    change_weights = error * sigmoid_derivative(hidden_output)
    weights += np.dot(inputs.T, change_weights)

# Тестирование нейронной сети
test_input = np.array(
    [[0, 1, 0],
     [0, 1, 0],
     [0, 1, 0]]
)

prediction_input = np.dot(test_input, weights)
prediction_output = sigmoid(prediction_input)

print(prediction_output)
