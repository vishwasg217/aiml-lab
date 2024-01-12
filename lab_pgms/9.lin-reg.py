import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 1000)
y = np.log(np.abs((x ** 2) - 1) + 0.5)
x = x + np.random.normal(scale=0.05, size=1000) 
plt.scatter(x, y, alpha=0.3)
def local_regression(x0, x, y, tau): 
    x0 = [1, x0]   
    x = [[1, i] for i in x]
    x = np.asarray(x)
    xw = (x.T) * np.exp(np.sum((x - x0) ** 2, axis=1) / (-2 * tau))
    beta = np.linalg.pinv(xw @ x) @ xw @ y 
    return x0 @ beta


def plot_lr(tau):
    domain = np.linspace(-5, 5, num=500)
    pred = [local_regression(x0, x, y, tau) for x0 in domain] 
    plt.scatter(x, y, alpha=0.3)
    plt.plot(domain, pred, color="red") 
    return plt


plot_lr(1).show()
