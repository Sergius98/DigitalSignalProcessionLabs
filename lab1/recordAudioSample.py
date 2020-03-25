import sounddevice as sd
from scipy.io.wavfile import write

fs = 44100  # Sample rate
seconds = 3  # Duration of recording
filename = 'audio.wav'

print("recording has started. duration = "+ str(seconds) + " seconds")
myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2, dtype='int16')
sd.wait()  # Wait until recording is finished
print("recording is finished")
write(filename, fs, myrecording)  # Save as WAV file
print("recording saved to " + filename)
