import sounddevice as sd
from scipy.io.wavfile import write


def make_audio(filename='audio.wav', seconds=3, fs=44100, audio_type='int16', nchannels=2):

    # fs = 44100  # Sample rate
    # seconds = 3  # Duration of recording
    # filename = 'audio.wav'
    print("recording has started. duration = "+ str(seconds) + " seconds")
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=nchannels, dtype=audio_type)
    sd.wait()  # Wait until recording is finished
    print("recording is finished")
    write(filename, fs, myrecording)  # Save as WAV file
    print("recording saved to " + filename)
