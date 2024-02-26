import numpy as np


class Perceptron:
    def __init__(self, num_features):
        self.weights = np.zeros(num_features)
        self.bias = 0.0

    def predict(self, x):
        return np.sign(np.dot(self.weights, x) + self.bias)

    def train(self, X_train, y_train, epochs=100, learning_rate=0.1):
        for _ in range(epochs):
            for i in range(len(X_train)):
                prediction = self.predict(X_train[i])
                if prediction != y_train[i]:
                    self.weights += learning_rate * y_train[i] * X_train[i]
                    self.bias += learning_rate * y_train[i]

    def __str__(self):
        return f"Weights: {self.weights}, Bias: {self.bias}"


# Генерация данных
X_train = np.random.uniform(0, 1, (20, 2))
y_train = np.array([1 if x[0] > x[1] else -1 for x in X_train])


# Обучение перцептрона
perceptron = Perceptron(2)
perceptron.train(X_train, y_train)
print(perceptron)


# Тестирование на 1000 точках
X_test = np.random.uniform(0, 1, (1000, 2))
y_test = np.array([1 if x[0] > x[1] else -1 for x in X_test])


correct_predictions = 0
for i in range(len(X_test)):
    if perceptron.predict(X_test[i]) == y_test[i]:
        correct_predictions += 1

accuracy = correct_predictions / len(X_test)
print(f"Точность на 1000 точках: {accuracy}")
