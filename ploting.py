import matplotlib.pyplot as plt


def plot2(title, first_list, first_name, second_list, second_name):
    plt.subplot(2, 1, 1)
    plt.plot(first_list, 'b')
    plt.title(title)
    plt.ylabel(first_name)
    plt.subplot(2, 1, 2)
    plt.plot(second_list, 'r')
    plt.ylabel(second_name)
    plt.show()