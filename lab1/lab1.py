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


# in case we need large fragments for the task, not the small ones
# make n or n+1 slices (if there's no way to equally delete it)
def make_slices(arr, at_least_n):
    n = at_least_n
    narr = []
    length = len(arr)
    nlength = length // n
    for i in range(0, n):
        narr.append(arr[i * nlength:(i + 1) * nlength])
    diff = length - n * nlength
    if diff >= nlength // 2:
        narr.append(arr[n * nlength:length])
    elif diff > 0:
        for i in range(length - diff - 1, length):
            narr[-1].append(arr[i])
    return narr


# print(make_slices([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14], 2))
# print(make_slices([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14], 3))
# print(make_slices([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14], 4))
# print(make_slices([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14], 5))
# print(make_slices([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14], 6))
# exit()


def fragmented_fft_fragmented_audio(fragmented_audio, n):
    sliced_fft = []
    sliced_audio = make_slices(fragmented_audio, n)
    for i in range(0, len(sliced_audio)):
        sliced_fft.append(fft_fragmented_audio(sliced_audio[i]))
    return sliced_fft


def show_sliced(filename="audio.wav", n=100):
    audio_from_file = read_audio(filename)
    audio_as_array = fragment_audio(audio_from_file)
    transformed_sliced_fragmented_audio = fragmented_fft_fragmented_audio(audio_as_array, n)
    united_fft = np.array([])
    for fragment in transformed_sliced_fragmented_audio:
        united_fft = np.concatenate((united_fft, fragment), axis=None)
    # print(len(transformed_sliced_fragmented_audio[0]))
    # print(len(transformed_sliced_fragmented_audio[-1]))
    # print(len(united_fft))
    # exit()
    plot2(filename, np.asarray(audio_as_array), 'fragmented audio',
          united_fft, 'FFT fragmented audio')


def show(filename="audio.wav"):
    audio_from_file = read_audio(filename)
    audio_as_array = fragment_audio(audio_from_file)
    trimmed_part = len(audio_as_array) // 20000  # to make plot prettier
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
          np.asarray(audio_as_array[audio_length // 10:audio_length - audio_length // 10]),
          'fragmented audio',
          transformed_fragmented_audio[audio_length // 10:audio_length - audio_length // 10],
          'FFT fragmented audio')
    # plot2(audio_filename, abs(np.asarray(audio_as_array)), 'fragmented audio',
    #       abs(transformed_fragmented_audio), 'FFT fragmented audio')
    # plot2(audio_filename + ' : zoom in',
    #       abs(np.asarray(audio_as_array[audio_length // 10:audio_length - audio_length // 10])),
    #       'fragmented audio',
    #       abs(transformed_fragmented_audio[audio_length // 10:audio_length - audio_length // 10]),
    #       'FFT fragmented audio')


# show(filename="../lab2/audio2.wav")
# show()
show_sliced()
