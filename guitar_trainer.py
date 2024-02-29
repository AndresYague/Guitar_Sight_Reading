import pyaudio
import matplotlib.pyplot as plt
import numpy as np
import sys

NOTES = {
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

POSITIONS = {
        "mi4": -9,
        "fa4": -8,
        "sol4": -7,
        "la3": -6,
        "si3": -5,
        "do3": -4,
        "re3": -3,
        "mi3": -2,
        "fa3": -1,
        "sol3": 0,
        "la2": 1,
        "si2": 2,
        "do2": 3,
        "re2": 4,
        "mi2": 5,
        "fa2": 6,
        "sol2": 7,
        "la1": 8,
        "si1": 9,
        "do1": 10,
        "re1": 11,
        "mi1": 12,
        }

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

def paint_note(position, current=12):
    '''
    Paint the chosen note to the terminal
    '''

    s = ""

    # At this point we are done
    if current < -9:
        return s

    if current > 7 and position < current:
        return paint_note(position, current - 1)

    if current < -3 and position > current:
        return s

    blank = ' '
    line = '-'
    note = 'o'
    breakln = '\n'

    def blank_note():
        return 13 * blank + note + breakln
    def line_note_extra():
        return 10 * blank + 3 * line + note + 3 * line + breakln
    def line_note():
        return 13 * line + note + 13 * line + breakln
    def line_empty_extra():
        return 10 * blank + 7 * line + breakln
    def line_empty():
        return 27 * line + breakln

    # Start building from the top down.
    # The highest possible position is
    # +12, but we only need to put extra lines
    # up top from +8 (la).

    if current == 0:
        s = "S"

    if current == position:
        if current > 7 or current < -3:
            if (current % 2) == 0:
                s += line_note_extra()
            else:
                s += blank_note()
        else:
            if (current % 2) == 0:
                s += line_note()
            else:
                s += blank_note()
    else:
        if (current % 2) == 0:
            if current > 7 or current < -3:
                s += line_empty_extra()
            else:
                s += line_empty()
        else:
            s += breakln

    s += paint_note(position, current - 1)
    return s

def main():
    '''
    Train playing the guitar from sheet music
    '''

    n_notes = 10
    record_seconds = 10

    for _ in range(n_notes):

        # Choose random note
        note_name = np.random.choice(list(NOTES.keys()))
        note_tone = NOTES[note_name]

        # Paint the chosen note to the terminal
        print("Play this note:")
        s = paint_note(POSITIONS[note_name])
        print(s)

        is_found = find_frequency(note_freq(note_tone), n_consecutive=8,
                                  record_seconds=record_seconds)

        # Feedback
        if is_found:
            s = "That's right!"
        else:
            s = "Sorry, that was not the note"
        print(s)

if __name__ == "__main__":
    main()
