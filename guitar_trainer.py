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


def find_frequency(freq, n_consecutive=5, record_seconds=5, chunk=1024,
                   chunkps=2, debug=False):
    '''
    Keep recording until freq is found a n_consecutive number of times,
    consecutively

    record_seconds is the maximum record time
    chunk is the number of byte strings to get
    chunksps is how many chunks in a second to record
    '''

    # Define the tolerance of quarter tone
    tol_p = 2 ** (1 / 24)
    tol_m = 2 ** (-1 / 24)

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
            if debug:
                dfreq = frequency[1] - frequency[0]
                s = f"dt = {dt:.2e} s; dfreq = {dfreq:.2e} Hz"
                print(s)

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

        if debug:
            plt.plot(frequency, spectrum_norm, "o-")
            plt.xlim(0, 1000)
            plt.show()

        if debug:
            s = f"max_freq = {max_freq:.2f} Hz; max_freq2 = {max_freq2:.2f} Hz"
            s += f" objective frequency = {freq:.2f} Hz"
            print(s)

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

    if debug:
        print(tol_p * freq, freq, tol_m * freq)

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
    put_note = current == position

    # Start building from the top down.
    # The highest possible position is
    # +12, but we only need to put extra lines
    # up top from +8 (la).

    # Define the functions that will paint the note
    def line_extra(put_note):
        character = note if put_note else line
        return 10 * blank + 3 * line + character + 3 * line + breakln

    def line_normal(put_note):
        character = note if put_note else line
        return 13 * line + character + 13 * line + breakln

    def line_blank(put_note):
        character = note if put_note else blank
        return 13 * blank + character + breakln

    if current == 0:
        s = "S"

    if current > 7 or current < -3:
        if (current % 2) == 0:
            s += line_extra(put_note)
        else:
            s += line_blank(put_note)
    else:
        if (current % 2) == 0:
            s += line_normal(put_note)
        else:
            s += line_blank(put_note)

    s += paint_note(position, current - 1)
    return s


def main():
    '''
    Train playing the guitar from sheet music
    '''

    n_notes = 10
    record_seconds = 10
    debug_notes = None

    if debug_notes is None:
        notes_choice = list(NOTES.keys())
    else:
        notes_choice = debug_notes

    for _ in range(n_notes):

        # Choose random note
        note_name = np.random.choice(notes_choice)
        note_tone = NOTES[note_name]

        # Paint the chosen note to the terminal
        print("Play this note:")
        s = paint_note(POSITIONS[note_name])
        print(s)

        # Check for debugging
        if debug_notes is None:
            debug = False
        elif note_name in debug_notes:
            debug = True

        is_found = find_frequency(note_freq(note_tone), n_consecutive=3,
                                  record_seconds=record_seconds, debug=debug)

        # Feedback
        if is_found:
            s = "That's right!"
        else:
            s = "Sorry, that was not the note"
        print(s)


if __name__ == "__main__":
    main()
