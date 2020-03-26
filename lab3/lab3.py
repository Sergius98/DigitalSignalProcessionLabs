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

silence = silence.detect_silence(myaudio, min_silence_len=1000, silence_thresh=-16)

silence = [((start/1000),(stop/1000)) for start,stop in silence]  # convert to sec
print(silence)

