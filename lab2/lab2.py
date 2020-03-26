import librosa

filename = "audio2.wav"
input, sample_rate = librosa.load(filename)
mfcc_sequence = librosa.feature.mfcc(y=input, sr=sample_rate)
