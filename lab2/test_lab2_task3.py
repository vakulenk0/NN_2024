import numpy as np


# Функции активации
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# Обучающая матрица
X = np.array([[0, 1, 0, 0, 1, 0, 0, 1, 0]])


# Ожидаемые результаты
y = np.array([[0, 0, 0, 1],
              [0, 0, 1, 0],
              [0, 1, 0, 0],
              [1, 0, 0, 0]])


# Инициализация весов сети
input_neurons = 9
hidden_neurons = 5
output_neurons = 4

weights_input_hidden = np.random.uniform(size=(input_neurons, hidden_neurons))
weights_hidden_output = np.random.uniform(size=(hidden_neurons, output_neurons))

# Обучение сети
epochs = 1
learning_rate = 0.1

for epoch in range(epochs):
    # Считаем индуцированные локальные поля для нейронов входного
    # и выходного слоёв и прогоняем их через функцию активации
    hidden_layer_ind_pole_neuron = np.dot(X, weights_input_hidden)
    hidden_layer_output = sigmoid(hidden_layer_ind_pole_neuron)

    print(hidden_layer_ind_pole_neuron)
    print(hidden_layer_output)
    print("\n")


    output_layer_input = np.dot(
        hidden_layer_output,
        weights_hidden_output
    )

    output = sigmoid(output_layer_input)
    print(output_layer_input)
    print(output)


    # Обратное распространение
    error = y - output
    print("\n")
    print(error[0])
    # output_gradient = sigmoid_derivative(output)
    # output_delta = error * output_gradient
    #
    # error_hidden = output_delta.dot(weights_hidden_output.T)
    # hidden_gradient = sigmoid_derivative(hidden_layer_output)
    # hidden_delta = error_hidden * hidden_gradient
    #
    # # Обновление весов
    # weights_hidden_output += hidden_layer_output.T.dot(output_delta) * learning_rate
    # weights_input_hidden += X.T.dot(hidden_delta) * learning_rate


# Предсказание
def predict(input_data):
    hidden_layer_input = np.dot(input_data, weights_input_hidden)
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
    output = sigmoid(output_layer_input)

    return output


# Тестирование сети
test_data_X = np.array([[1, 0, 1],
                         [0, 1, 0],
                         [1, 0, 1]])

test_data_Y = np.array([[1, 0, 1],
                         [0, 1, 0],
                         [0, 1, 0]])

test_data_L = np.array([[1, 0, 0],
                         [1, 0, 0],
                         [1, 1, 1]])

test_data_I = np.array([[0, 1, 0],
                         [0, 1, 0],
                         [0, 1, 0]])

# print(predict(test_data_X))
# print(predict(test_data_Y))
# print(predict(test_data_L))
# print(predict(test_data_I))
