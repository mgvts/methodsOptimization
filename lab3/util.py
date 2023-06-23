from matplotlib import pyplot as plt


def show_image(X, Y):
    fig, ax = plt.subplots(1)

    plt.plot(X[:], Y[:], 'o', label='data')

    ax.set_xlim([0, 15])
    ax.set_ylim([-10, 10])

    plt.show()


def derivative(f, x, method='central', h=0.01):
    if method == 'central':
        return (f(x + h) - f(x - h)) / (2 * h)
    elif method == 'forward':
        return (f(x + h) - f(x)) / h
    elif method == 'backward':
        return (f(x) - f(x - h)) / h
    else:
        raise ValueError("Unknown differentiation method")
