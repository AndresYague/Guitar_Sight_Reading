import pyaudio
import matplotlib.pyplot as plt
import numpy as np
import sys

def find_frequency(freq, n_consecutive=5, record_seconds=5, chunk=2048,
                   chunkps=16):
    '''
    Keep recording until freq is found a n_consecutive number of times,
    consecutively

    record_seconds is the maximum record time
    chunk is the number of byte strings to get
    chunksps is how many chunks in a seconds to record
    '''

    # Define the tolerance of quarter tone
    tol_p = 2 ** (1 / 18)
    tol_m = 2 ** (-1 / 18)

    # Take 32 bits for high amplitudes
    format_ = pyaudio.paInt32
    channels = 1 if sys.platform == 'darwin' else 2

    # Calculate the byte rate
    rate = chunkps * chunk

    # Open the stream
    p = pyaudio.PyAudio()
    stream = p.open(format=format_, channels=channels, rate=rate, input=True)

    # Variables for fft
    time = 1 / chunkps
    total_time = 0
    dt = None

    # Record
    found_consecutive = 0
    did_found = False
    while True:

        # Read the amplitude in byte strings
        amplitude = stream.read(chunk)

        # Transform to int32
        amplitude = np.frombuffer(amplitude, dtype="int32")

        # Define dt if not defined already
        if dt is None:
            dt = time / amplitude.shape[0]
            frequency = np.fft.fftfreq(amplitude.shape[0], d=dt)

        # Perform the fft and take the norm of it
        spectrum = np.fft.fft(amplitude)
        spectrum_norm = np.sqrt(spectrum.real ** 2 + spectrum.imag ** 2)

        # Find the index of the highest frequency component
        index = np.argmax(spectrum_norm)

        # Also find the next highest frequency
        try:
            index2 = np.argmax(spectrum_norm[:index - 1])
        except ValueError:
            index2 = 0

        # Retrieve the values associated with the index
        max_freq = np.abs(frequency[index])
        max_freq2 = np.abs(frequency[index2])
        max_spec = spectrum_norm[index]

        # Check the frequency ratio
        ratio = max_freq / freq
        ratio2 = max_freq2 / freq

        # Check if the ratio is within the tolerances
        comp1 = ratio > tol_m and ratio < tol_p
        comp2 = ratio2 > tol_m and ratio2 < tol_p

        if comp1 or comp2:
            found_consecutive += 1
        else:
            found_consecutive = 0

        # If found enough consecutives, we can assume the frequency
        # was being played
        if found_consecutive >= n_consecutive:
            did_found = True
            break

        # If time exceeds recording time, break
        total_time += time
        if total_time > record_seconds:
            break

    stream.close()
    p.terminate()

    return did_found

def note_freq(halftone_from_la, lahz=440):
    '''
    Give the approximated frequency of a named note
    '''

    # From this "la" to the next one, the frequency doubles
    freq = 2 ** (halftone_from_la / 12) * lahz

    return freq

def create_tone_dict():
    '''
    Create the frequency dictionary from names to halftones from la
    '''

    notes = {
            "mi4": -29,
            "fa4": -28,
            "sol4": -26,
            "la3": -24,
            "si3": -22,
            "do3": -21,
            "re3": -19,
            "mi3": -17,
            "fa3": -16,
            "sol3": -14,
            "la2": -12,
            "si2": -10,
            "do2": -9,
            "re2": -7,
            "mi2": -5,
            "fa2": -4,
            "sol2": -2,
            "la1": 0,
            "si1": 2,
            "do1": 3,
            "re1": 5,
            "mi1": 7,
            }

    return notes

def main():
    '''
    Train playing the guitar from sheet music
    '''

    n_notes = 10
    record_seconds = 10
    notes = create_tone_dict()

    for _ in range(n_notes):

        # Choose random note
        note_name = np.random.choice(list(notes.keys()))
        note_tone = notes[note_name]

        print(note_name)

        is_found = find_frequency(note_freq(note_tone), n_consecutive=8,
                                  record_seconds=record_seconds)

        print(is_found)

    halftones = -14
    is_found = find_frequency(note_freq(halftones), n_consecutive=5,
                              record_seconds=record_seconds)

if __name__ == "__main__":
    main()
