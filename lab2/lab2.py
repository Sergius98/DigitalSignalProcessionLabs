import librosa
from ploting import plot1, plot_mfccs


def read_audio(filename):

    input_audio, sample_rate = librosa.load(filename)

    return input_audio, sample_rate


def mfcc(input_audio, sample_rate, num_mfcc):

    mfcc_sequence = librosa.feature.mfcc(y=input_audio, sr=sample_rate, n_mfcc=num_mfcc)

    return mfcc_sequence


def show(filename="audio2.wav"):
    # read audio from file
    input_audio, sample_rate = read_audio(filename)

    # draw audio on a plot
    plot1(filename, input_audio, 'input audio')

    # make mfccs
    mfcc_sequence = mfcc(input_audio, sample_rate, 12)

    # draw mfcc on a plot
    plot_mfccs(mfcc_sequence, title='Mel-cepstrum coefficients',
               filename=filename)
    # mfcc_sequence = librosa.feature.mfcc(y=input_audio, sr=sample_rate, n_mfcc=40)
    # plot_mfccs(mfcc_sequence, title='Mel-cepstrum coefficients',
    #            filename=filename)
    # mfcc_sequence = librosa.feature.mfcc(y=input_audio, sr=sample_rate, dct_type=2)
    # plot_mfccs(mfcc_sequence, title='Mel-cepstrum coefficients',
    #            filename=filename)
    # mfcc_sequence = librosa.feature.mfcc(y=input_audio, sr=sample_rate, dct_type=3)
    # plot_mfccs(mfcc_sequence, title='Mel-cepstrum coefficients',
    #            filename=filename)


# show(filename="../lab1/audio.wav")
show()
