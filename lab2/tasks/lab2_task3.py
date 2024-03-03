import random

import numpy as np

letters = ['X', 'Y', 'I', 'L']
# Задаем входной вектор X, Y, L или I
examples = np.array([[1, 0, 1, 0, 1, 0, 1, 0, 1],
              [1, 0, 1, 0, 1, 0, 0, 1, 0],
              [0, 1, 0, 0, 1, 0, 0, 1, 0],
              [1, 0, 0, 1, 0, 0, 1, 1, 1]])

# Задаем выходные метки для каждой из букв X, Y, I, L
# X: [0, 0, 0, 1], Y: [0, 0, 1, 0], I: [0, 1, 0, 0], L: [1, 0, 0, 0]
need_outputs = np.array(
             [[0, 0, 0, 1],
             [0, 0, 1, 0],
             [0, 1, 0, 0],
             [1, 0, 0, 0]])

input_neurons = 9
hidden_neurons = 5
output_neurons = 4

weights_input_hidden = np.random.uniform(size=(input_neurons, hidden_neurons))
weights_hidden_output = np.random.uniform(size=(hidden_neurons, output_neurons))
bias_hidden = np.random.uniform(size=(1, hidden_neurons))
bias_output = np.random.uniform(size=(1, output_neurons))

# Задаем скорость обучения и количество эпох
learning_rate = 0.01
epochs = 1000


# Функция активации (сигмоида)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Производная функции активации
def sigmoid_derivative(x):
    return x * (1 - x)

# Обучение нейронной сети
for i in range(len(examples)):
    for epoch in range(epochs):
        # Forward pass
        hidden_layer_input = np.dot(examples[i], weights_input_hidden) + bias_hidden
        hidden_layer_output = sigmoid(hidden_layer_input)

        output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
        output = sigmoid(output_layer_input)

        # Backpropagation
        error = need_outputs[i] - output
        d_output = error * sigmoid_derivative(output)

        error_hidden = d_output.dot(weights_hidden_output.T)
        d_hidden = error_hidden * sigmoid_derivative(hidden_layer_output)

        # Обновление весов и смещений
        weights_hidden_output += hidden_layer_output.T.dot(d_output) * learning_rate
        bias_output += np.sum(d_output, axis=0, keepdims=True) * learning_rate

        weights_input_hidden += examples[i].reshape(9, 1).dot(d_hidden) * learning_rate
        bias_hidden += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

    test_input = np.array(examples[i])
    hidden_layer_input = np.dot(test_input, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    predicted_output = sigmoid(output_layer_input)

    # Выводим предсказанный выход
    print(predicted_output)

print()
print("////////////////////////////////////////////\n")

correct_prediction = 0

for i in range(1000):
    n = random.randint(0,3)
    test_input = np.array(examples[n])  # Тестовый вход для буквы X
    hidden_layer_input = np.dot(test_input, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    predicted_output = sigmoid(output_layer_input)
    arr = []
    for elem in predicted_output:
        arr.append(elem)
    if arr.index(max(arr)) == n:
        correct_prediction += 1

print("Accuracy: " + str(100 - float(correct_prediction/1000)*100) + str("%"))


