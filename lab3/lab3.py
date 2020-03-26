import os

# set path before you start
# download ffmpeg
# set PATH=%PATH%;C:\Users\BIGse\git\DigitalSignalProcessionLabs\lab3\ffmpeg\bin\
# ";C:\\Users\\BIGse\\git\\DigitalSignalProcessionLabs\\lab3\\ffmpeg\\bin\\"
os.environ['PATH'] = os.environ['PATH'] + ";" + os.path.abspath("ffmpeg/bin/")
print(os.environ['PATH'])

from pydub import AudioSegment, silence


filename = "audio3.wav"
min_silence_len = 50
silence_thresh = -32


def open_audio(file_name="audio3.wav"):
    audio = AudioSegment.from_wav(file_name)
    return audio


def get_pauses(audio, min_silence_length=50, silence_threshold=-32):
    pauses = silence.detect_silence(audio, min_silence_len=min_silence_length, silence_thresh=silence_threshold)

    pauses = [((start / 1000), (stop / 1000)) for start, stop in pauses]  # convert to sec

    return pauses


def get_words(audio, min_silence_length=50, silence_threshold=-32):
    audio_chunks = silence.split_on_silence(audio,
                                            # must be silent for at least 50ms
                                            min_silence_len=min_silence_length,

                                            # consider it silent if quieter than -32 or -30 dBFS
                                            silence_thresh=silence_threshold
                                            )
    return audio_chunks


def output_words(audio_words):
    for i, chunk in enumerate(audio_words):
        out_file = ".//splitAudio//word{0}.wav".format(i)
        print("exporting", out_file)
        chunk.export(out_file, format="wav")


def show(file_name, min_silence_length, silence_threshold):
    audio = open_audio()
    pauses = get_pauses(audio)
    words = get_words(audio)
    print(pauses)
    output_words(words)


show(filename, min_silence_len, silence_thresh)


