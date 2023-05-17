from matplotlib import pyplot as plt


def show_image(X, Y):
    fig, ax = plt.subplots(1)

    plt.plot(X[:], Y[:], 'o', label='data')

    ax.set_xlim([0, 15])
    ax.set_ylim([-10, 10])

    plt.show()