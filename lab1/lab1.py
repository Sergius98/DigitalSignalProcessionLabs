import wave

filename = "audio.wav"

# read audio from file
def read_audio(filename):
    audio = wave.open(filename, 'rb')
    return audio


# fragment audio
def fragment_audio(audio):
    pass


# Fast Fourier Transform for the fragmented audio
def fft_fragmented_audio(fragmented_audio):
    pass


read_audio(filename)

