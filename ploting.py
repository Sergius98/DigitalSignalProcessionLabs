import librosa.display
import matplotlib.pyplot as plt


def plot1(title, list, name):
    plt.plot(list, 'b')
    plt.title(title)
    plt.ylabel(name)
    plt.show()


def plot2(title, first_list, first_name, second_list, second_name):
    plt.subplot(2, 1, 1)
    plt.plot(first_list, 'b')
    plt.title(title)
    plt.ylabel(first_name)
    plt.subplot(2, 1, 2)
    plt.plot(second_list, 'r')
    plt.ylabel(second_name)
    plt.show()


def plot_mfccs(mfccs, title='MFCC', filename=''):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, x_axis='time')
    plt.colorbar()
    plt.title(title + " : " + filename)
    plt.tight_layout()
    plt.show()
