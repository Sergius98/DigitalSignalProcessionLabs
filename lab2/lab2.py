import librosa
from ploting import plot1, plot_mfccs


def show(filename="audio2.wav"):
    input_audio, sample_rate = librosa.load(filename)
    mfcc_sequence = librosa.feature.mfcc(y=input_audio, sr=sample_rate)
    plot1(filename, input_audio, 'input audio')
    plot_mfccs(mfcc_sequence, title='Mel-cepstrum coefficients',
               filename=filename)


show(filename="../lab1/audio.wav")
#show()
