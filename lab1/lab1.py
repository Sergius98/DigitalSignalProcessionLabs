from scipy.fftpack import fft
from scipy.io import wavfile
from ploting import plot2
import numpy as np


# read audio from file
def read_audio(filename):
    fs, audio = wavfile.read(filename)  # load the data
    return audio


# fragment audio
def fragment_audio(audio):
    if type(audio.T[0]) == np.ndarray:
        fragmented_audio = audio.T[0]  # this is a two or more channel soundtrack, I get the first track
    else:
        fragmented_audio = audio.T
    return fragmented_audio


# Fast Fourier Transform for the fragmented audio
def fft_fragmented_audio(fragmented_audio):

    # normalized_fragmented_audio = [(ele / 2 ** 16.) * 2 - 1 for ele in fragmented_audio]  # this is 16-bit track, b is now normalized on [-1,1)
    # print(normalized_fragmented_audio)
    # transformed_fragmented_audio = fft(normalized_fragmented_audio)
    # transformed_fragmented_audio = fft(fragmented_audio[1:])
    # print(transformed_fragmented_audio)
    # r = (len(transformed_fragmented_audio) // 1) - 1  # right limit for ploting
    # print(str(len(normalized_fragmented_audio)) + ":" + str(len(transformed_fragmented_audio)))
    # plt.subplot(2, 1, 1)
    # plt.plot(abs(np.asarray(fragmented_audio)), 'b')
    # plt.title('audio file')
    # plt.ylabel('fragmented audio')
    # plt.subplot(2, 1, 2)
    # plt.plot(abs(transformed_fragmented_audio), 'r')
    # plt.ylabel('Fast Fourier Transform fragmented audio')
    # plt.ylabel('FFT fragmented audio')
    # plt.figure()
    return fft(fragmented_audio)


def show(filename="audio.wav"):
    audio_from_file = read_audio(filename)
    audio_as_array = fragment_audio(audio_from_file)
    trimmed_part = len(audio_as_array)//20000  # to make plot prettier
    if trimmed_part > 100:
        trimmed_part = 100
    elif trimmed_part < 9 and len(audio_as_array) > 100:
        trimmed_part = 9
    print(trimmed_part)
    audio_as_array = audio_as_array[trimmed_part:]
    audio_length = len(audio_as_array)
    transformed_fragmented_audio = fft_fragmented_audio(audio_as_array)
    plot2(filename, np.asarray(audio_as_array), 'fragmented audio',
          transformed_fragmented_audio, 'FFT fragmented audio')
    plot2(filename + ' : zoom in',
          np.asarray(audio_as_array[audio_length//10:audio_length - audio_length//10]),
          'fragmented audio',
          transformed_fragmented_audio[audio_length//10:audio_length - audio_length//10],
          'FFT fragmented audio')
    # plot2(audio_filename, abs(np.asarray(audio_as_array)), 'fragmented audio',
    #       abs(transformed_fragmented_audio), 'FFT fragmented audio')
    # plot2(audio_filename + ' : zoom in',
    #       abs(np.asarray(audio_as_array[audio_length // 10:audio_length - audio_length // 10])),
    #       'fragmented audio',
    #       abs(transformed_fragmented_audio[audio_length // 10:audio_length - audio_length // 10]),
    #       'FFT fragmented audio')


# show(filename="../lab2/audio2.wav")
show()

