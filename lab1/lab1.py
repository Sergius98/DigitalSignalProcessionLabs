#import wave
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.io import wavfile
import numpy as np

audio_filename = "audio.wav"


# read audio from file
def read_audio(filename):
    # audio = wave.open(filename, 'rb')
    fs, audio = wavfile.read(audio_filename)  # load the data
    return audio


# fragment audio
def fragment_audio(audio):
    fragmented_audio = audio.T[0]  # this is a two channel soundtrack, I get the first track
    # print(fragmented_audio)
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


audio_from_file = read_audio(audio_filename)
audio_as_array = fragment_audio(audio_from_file)
audio_as_array = audio_as_array[9:]
audio_length = len(audio_as_array)
# audio_as_array = audio_as_array[audio_length//20:audio_length//2]
# audio_as_array = audio_as_array[audio_length//20:audio_length//20 + 100]
transformed_fragmented_audio = fft_fragmented_audio(audio_as_array)
plt.subplot(2, 1, 1)
# plt.plot(abs(np.asarray(audio_as_array[audio_length//20:audio_length//2])), 'b')
plt.plot(abs(np.asarray(audio_as_array)), 'b')
plt.title('audio file')
plt.ylabel('fragmented audio')
plt.subplot(2, 1, 2)
# plt.plot(abs(transformed_fragmented_audio[audio_length//20:audio_length//2]), 'r')
plt.plot(abs(transformed_fragmented_audio), 'r')
plt.ylabel('FFT fragmented audio')

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(abs(np.asarray(audio_as_array[audio_length//10:audio_length - audio_length//10])), 'b')
plt.title('zoom in')
plt.ylabel('fragmented audio')
plt.subplot(2, 1, 2)
plt.plot(abs(transformed_fragmented_audio[audio_length//10:audio_length - audio_length//10]), 'r')
plt.ylabel('FFT fragmented audio')

# plt.figure()
# normalized_fragmented_audio = [(ele / 2 ** 16.) * 2 - 1 for ele in audio_as_array]  # this is 16-bit track, b is now normalized on [-1,1)
# transformed_normalized_fragmented_audio = fft_fragmented_audio(normalized_fragmented_audio)
# plt.subplot(2, 1, 1)
# plt.plot(abs(np.asarray(normalized_fragmented_audio)), 'b')
# plt.title('normalized input')
# plt.ylabel('normalized audio')
# plt.subplot(2, 1, 2)
# plt.plot(abs(transformed_normalized_fragmented_audio), 'r')
# plt.ylabel('FFT normalized audio')

audio_length = len(audio_as_array)
small_audio_as_array = audio_as_array[audio_length//20:audio_length//2]
plt.show()


