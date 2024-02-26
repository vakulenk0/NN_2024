import random


class Dot:
    def __init__(self):
        self.coord = [
            random.randint(0,1),
            random.randint(0,1)
        ]

    def calculate(self, w: list):
        return max(0, self.coord[0] * w[0] + self.coord[1] * w[1]) % 2

    def out(self):
        return self.coord


class NeuroNet:

    def __init__(self):
        self.w = [1, 1]

        self.dots = [Dot() for i in range(10)]

    def start(self):
        for i in range(10):
            if self.dots[i].calculate(self.w) == 1:
                print(self.dots[i].out())
                print("Точка принадлежит к 2ому классу\n")
            else:
                print(self.dots[i].out())
                print("Точка принадлежит к 1ому классу\n")


def main():
    n = NeuroNet()
    n.start()


if __name__ == "__main__":
    main()
