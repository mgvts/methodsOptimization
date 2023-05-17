from matplotlib import pyplot as plt


def show_image(X, Y):
    fig, ax = plt.subplots(1)

    plt.plot(X[:], Y[:], 'o', label='data')

    ax.set_xlim([-30, 30])
    ax.set_ylim([-30, 30])

    plt.show()