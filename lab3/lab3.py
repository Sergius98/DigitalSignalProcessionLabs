import os

# set path before you start
# download ffmpeg
# set PATH=%PATH%;C:\Users\BIGse\git\DigitalSignalProcessionLabs\lab3\ffmpeg\bin\
# ";C:\\Users\\BIGse\\git\\DigitalSignalProcessionLabs\\lab3\\ffmpeg\\bin\\"
os.environ['PATH'] = os.environ['PATH'] + ";" + os.path.abspath("ffmpeg/bin/")
print(os.environ['PATH'])

from pydub import AudioSegment,silence


# os.path.abspath("ffmpeg/bin/ffmpeg.exe")
# AudioSegment.converter=os.path.abspath("ffmpeg/bin/ffmpeg.exe")
myaudio = intro = AudioSegment.from_wav("audio3.wav")

pause = silence.detect_silence(myaudio, min_silence_len=50, silence_thresh=-32)

pause = [((start/1000),(stop/1000)) for start,stop in pause]  # convert to sec
print(pause)

audio_chunks = silence.split_on_silence(intro,
    # must be silent for at least 50ms
    min_silence_len=50,

    # consider it silent if quieter than -32 or -30 dBFS
    silence_thresh=-32
)
for i, chunk in enumerate(audio_chunks):

    out_file = ".//splitAudio//chunk{0}.wav".format(i)
    print ("exporting", out_file)
    chunk.export(out_file, format="wav")
