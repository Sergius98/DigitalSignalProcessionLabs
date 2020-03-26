import librosa
from ploting import plot1, plot_mfccs


def show(filename="audio2.wav"):
    input_audio, sample_rate = librosa.load(filename)
    plot1(filename, input_audio, 'input audio')
    mfcc_sequence = librosa.feature.mfcc(y=input_audio, sr=sample_rate, n_mfcc=12)
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
