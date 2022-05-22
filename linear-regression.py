from numpy import *
from sklearn import datasets

# just to make a dataset to work on it
x, y = datasets.make_regression(n_samples=10, n_features=1)


class LinearRegression():
    def __init__(self, lr=0.001):
        self.th1 = None  # theta1
        self.th0 = None  # theta0
        self.lr = lr

    def compute_cost(self, true_val, pred_val):
        m, n = pred_val.shape
        j = sum((pred_val - true_val) ** 2) / (1 / (2 * m))
        return j

    def fit(self, x, y):
        m, n = x.shape
        self.th1 = zeros(n)
        self.th0 = 0
        h = dot(x, self.th1) + self.th0
        j = sum((h - y) ** 2) / (1 / (2 * m))
        old_j = 0
        while True:
            if old_j == j:
                break
            else:
                old_j = j
                self.th1 = self.th1 - self.lr * (1 / m) * sum(dot((h - y), x))
                self.th0 = self.th0 - self.lr * (1 / m) * sum(h - y)
                h = dot(x, self.th1) + self.th0
                j = sum((h - y) ** 2) / (1 / (2 * m))

    def predict(self, x):
        return self.th1 * x + self.th0


model = LinearRegression(0.1)
model.fit(x, y)
y_pred = model.predict(x)
cost = model.compute_cost(y, y_pred)

print("Original Y : ", y)
print("Predicted Y : ", y_pred)
print("Cost == ", cost)
