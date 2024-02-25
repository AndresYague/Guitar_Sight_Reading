import pyaudio
import matplotlib.pyplot as plt
import numpy as np
import sys

def record_stream(record_seconds=5):
    '''
    Open a record stream with pyaudio
    '''

    chunk = 1024
    format_ = pyaudio.paInt32
    channels = 1 if sys.platform == 'darwin' else 2
    rate = 44100
    rate *= 2
    record_seconds = 5

    p = pyaudio.PyAudio()
    stream = p.open(format=format_, channels=channels, rate=rate, input=True)

    amplitude = []
    for _ in range(0, rate // chunk * record_seconds):
        value = stream.read(chunk)
        value = np.fromstring(value, np.int32)
        amplitude.append(value)
    amplitude = np.ndarray.flatten(np.array(amplitude))

    stream.close()
    p.terminate()

    time = np.linspace(0, record_seconds, amplitude.shape[0], endpoint=True)

    return time, amplitude

def main():
    '''
    Train playing the guitar from sheet music
    '''

    record_stream(record_seconds=2)

    pass

if __name__ == "__main__":
    main()
