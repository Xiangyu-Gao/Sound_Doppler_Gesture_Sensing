'''
Program: computer_Doppler_RADAR_v2.py
Author: Joao Nuno Carvalho
email:  joaonunocarv ( __AT__ ) gmail.com
Description: This program is a Doppler RADAR for a PC. In reality it’s a sonar that can determine the velocity
             of a moving object in relation to the PC speaker/mic when you put the speaker and mic near each
             other like in a laptop.
             This program is written in Python 3.5 and works in real-time using the PyAudio interface and NumPy.
             For this reason, in windows I have used the Anaconda 3.5 Distribution.

             This program is an implementation of the paper by Microsoft Research:

             SoundWave: Using the Doppler Effect to Sense Gestures
             Sidhant Gupta, Dan Morris, Shwetak Patel, Desney Tan
             Proceedings of ACM CHI 2012, May 2012
             http://research.microsoft.com/en-us/um/redmond/groups/cue/SoundWave/

             It has been inspired by a similar project that also implements this paper but for Javascript/Browser.
             https://github.com/DanielRapp/doppler

             To test this program, you should disable in the microphone settings, the echo cancelation and the
             noise reduction. In a small number of laptops, you must use an external microphone because the
             frequency response of those microphones has no gain at high frequencies, but this is rare.

             How the program works?
             This program makes two things at the same time, first it generates a sine wave of 18KHz that is
             outputted by the speakers and at the same time, it is processing the input of the microphone.
             All this action occurs in a callback function of the PyAudio frameworks that interfaces with the
             sound board of the computer. Each time the callback in called it receives a buffer with the
             microphone data of size 1024 and then it accumulates in a 2048 buffer, it then makes the FFT of
             that buffer and look for the amplitude of the signal at the frequencies around the 18KHz primary
             tone that we generate. If we move hour hand in the direction of the microphone we compress the
             sound wave, this means that there will be some amplitude in the frequencies upper to 18Khz,
             if we move hour hand away we will see some amplitude in the frequencies bellow the 18KHz.
             This is the Doppler effect and is why you hear a difference sound when a car passes through you
             pressing the horn. This is also the what police RADAR uses but that is with electromagnetic waves
             and this is also why we know the starts of distant galaxies are going away from us.
             That the universe is expanding. In the last case, the light of supernovas is becoming more reddish
             that it should be if they weren’t moving away. Light is also a electromagnetic wave.

             Going back to the program, then in the callback we put the result is a global variable doppler_shift
             and we copy the previous generated sin tone of 18KHz into the output buffer of the callback.

             In an external thread we periodically (0.5 seconds), see the value of this variable and print it
             to the screen as a text graph, at the end we plot the last FFT graph so we can see the signal.



The MIT License (MIT)

Copyright (c) 2016 Joao Nuno Carvalho

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of
the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''
import numpy as np


# This program only uses one channel (Mono).
def decode(in_data):
    # Convert the PyAudio mono byte stream [R0, R1, R2, ...] to a NumPy array.
    # Shape(chunk_length) instruction.
    result = np.fromstring(in_data, dtype=np.float32)
    return result


def encode(signal):
    # Convert a 1D numpy array into a mono byte stream for PyAudio.
    # Signal has chunk_length rows and one columns. It's a vector.
    out_data = signal.astype(np.float32).tostring()
    return out_data


def encode_int16(signal):
    # Convert a 1D numpy array into a mono byte stream for PyAudio.
    # Signal has chunk_length rows and one columns.
    # The output is scalded to a int16 value not a -1 to 1 value.
    signal_tmp = signal * ((2 ** 15) - 1)
    out_data = signal_tmp.astype(np.int16).tostring()
    return out_data
