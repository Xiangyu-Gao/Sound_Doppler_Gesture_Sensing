import pyaudio
import time
import numpy as np
import wave
import math
import matplotlib.pyplot as plt
from drawnow import drawnow
from utils import encode, decode, encode_int16


class Sine_tone:
    def __init__(self, sine_tone_buffer_size, SAMPLE_RATE, chunk_length, gen_freq):
        self.SAMPLE_RATE = SAMPLE_RATE
        self.chunk_length = chunk_length
        self.result = np.zeros((chunk_length, 1), dtype=np.float32)
        self.ptr_begin = 0
        self.frame_count_global = 0
        self.frames_to_file = []
        self.sine_tone_buffer_size = sine_tone_buffer_size
        self.sin_tone_buffer = np.zeros((sine_tone_buffer_size, 1), dtype=np.float32)
        self.sin_tone_buffer = self.fill_sin_tone_buffer(self.sin_tone_buffer, gen_freq)
        self.MIC_BUFFER_SIZE = chunk_length * 2
        self.mic_buffer = np.zeros(self.MIC_BUFFER_SIZE, dtype=np.float32)
        self.mic_buffer_cur_index = 0
        self.doppler_shift = (0, 0)
        self.fft_res = None
        self.relevant_freq_window = 33  # Relevant index window on the FFT, as described in the paper.
        self.fft_primary_tone_index = self.freqToFFTIndex(gen_freq)

    def fill_sin_tone_buffer(self, buffer, freq):
        amplitude = 1.0
        for i in range(0, len(buffer)):
            t = i / self.SAMPLE_RATE
            buffer[i] = amplitude * math.sin(2.0 * math.pi * freq * t)  # + teta initial phase ...... Wave equation
            # with sine.
        buffer[0] = 0.0

        return buffer

    def copy_sine_tone_buffer_to_output_buffer(self):
        # Copy sin buffer to speakers.
        ptr_end = self.ptr_begin + self.chunk_length
        if ptr_end < self.sine_tone_buffer_size:
            # copy the buffer in one time. ptr_begin and ptr_end
            np.copyto(self.result[0: self.chunk_length], self.sin_tone_buffer[self.ptr_begin: ptr_end])
            self.ptr_begin = self.ptr_begin + self.chunk_length
            if self.ptr_begin == self.sine_tone_buffer_size:
                self.ptr_begin = 0
        else:
            lacks = ptr_end - self.sine_tone_buffer_size
            dest_end = self.chunk_length - lacks
            # copy the first part of the buffer. ptr_begin and sine_tone_buffer_size
            np.copyto(self.result[0: dest_end], self.sin_tone_buffer[self.ptr_begin: self.sine_tone_buffer_size])
            dest_start = dest_end
            self.ptr_begin = 0
            # copy the second part of the buffer from the beginning. ptr_begin and lacks
            np.copyto(self.result[dest_start: self.chunk_length], self.sin_tone_buffer[self.ptr_begin: lacks])
            # Prepares for the next chunk filling
            self.ptr_begin = self.ptr_begin + lacks

        return self.result

    def freqToFFTIndex(self, freq):
        nyquist = self.SAMPLE_RATE / 2.0;
        return round((freq / nyquist) * (self.MIC_BUFFER_SIZE / 2.0))

    def copy_mic_chunk_to_mic_buffer(self, decoded_mic_data):
        '''
        Receives a decoded buffer in NumPy format and copies it to the mic_buffer that has two 2048 positions.
        It receives two buffers in two times and at the end, returns that it can calculate the FFT for
        the doppler shift.
        '''
        if self.mic_buffer_cur_index == 0:
            np.copyto(self.mic_buffer[0: self.chunk_length], decoded_mic_data[0: self.chunk_length])
            self.mic_buffer_cur_index = 1
            return False
        else:
            np.copyto(self.mic_buffer[self.chunk_length: self.chunk_length * 2], decoded_mic_data[0: self.chunk_length])
            self.mic_buffer_cur_index = 0
            return True

    def calc_doppler_direction(self):
        fft_data = np.abs(np.fft.fft(self.mic_buffer))

        # Obtain the bandwidth of the shifted signal that looks like a step function below the amplitude
        # of the primary tone and upper from the amplitude of the noise.
        primary_tone_volume = fft_data[self.fft_primary_tone_index]
        # This is an empirical ratio, find by experimentation but is equal to the one described in the paper 10%.
        max_volume_ratio = 0.1  # This ratio works
        # max_volume_ratio = 0.05;   # x20 lower than the signal value.

        left_bandwidth = 0
        last_left_bandwidth = 1

        while True:
            left_bandwidth += 1
            volume = fft_data[self.fft_primary_tone_index - left_bandwidth]
            normalized_volume = volume / primary_tone_volume
            if normalized_volume > max_volume_ratio:
                last_left_bandwidth = left_bandwidth
            if not (left_bandwidth < self.relevant_freq_window):
                break
        left_bandwidth = last_left_bandwidth

        right_bandwidth = 0
        last_right_bandwidth = 1
        while True:
            right_bandwidth += 1
            volume = fft_data[self.fft_primary_tone_index + right_bandwidth]
            normalized_volume = volume / primary_tone_volume
            if normalized_volume > max_volume_ratio:
                last_right_bandwidth = right_bandwidth
            if not (right_bandwidth < self.relevant_freq_window):
                break
        right_bandwidth = last_right_bandwidth

        # There are two threads working in this program, the communication between those
        # two threads is made by a global variable "doppler_shift".
        self.doppler_shift = (int(left_bandwidth), int(right_bandwidth))
        self.fft_res = np.fft.fftshift(fft_data)
        # return (left_bandwidth, right_bandwidth)

    def callback(self, in_data, frame_count, time_info, flag):
        # The callback receives the buffer of the input microphone and returns the buffer to output in the speakers.
        self.frame_count_global = frame_count

        # Processing the input data from microphone.
        decoded_mic_data = decode(in_data)
        flag_calc_Doppler_shift = self.copy_mic_chunk_to_mic_buffer(decoded_mic_data)
        # Detect the doppler effect, the shift in frequency around the central tone.
        if flag_calc_Doppler_shift:
            self.calc_doppler_direction()

        # Generating the output data to the speakers (continuous tone).
        self.copy_sine_tone_buffer_to_output_buffer()
        out_data = encode(self.result)

        # Collect the output generated and at the end writes a WAV file with them.
        self.frames_to_file.append(encode_int16(self.result))

        return out_data, pyaudio.paContinue

    def makeFig(self):
        fft_len = self.fft_res.shape[0]
        plt.plot(np.linspace(0, int(fft_len / 2) - 1, num=int(fft_len / 2)), self.fft_res[int(fft_len / 2):])
        plt.xlim([700, 900])


if __name__ == "__main__":
    CHANNELS = 1
    SAMPLE_RATE = 44100
    chunk_length = 1024     # Size of the buffer that the callback receives and gives to the PyAudio interface.
    gen_freq = 18000    # 18KHz, Frequency that we will be generating in the speakers.
    sine_tone_buffer_size = 44100  # The buffer as the size corresponding to 1 second.

    Sine_Dop = Sine_tone(sine_tone_buffer_size, SAMPLE_RATE, chunk_length, gen_freq)
    p = pyaudio.PyAudio()

    stream = p.open(format=pyaudio.paFloat32, channels=CHANNELS, rate=SAMPLE_RATE,
                    output=True,  # The callback will generate data to output to the speakers,
                    input=True,  # will also receive data from the microphone/input.
                    stream_callback=Sine_Dop.callback)  # The data is passed around in buffers of size chunk_length.
    stream.start_stream()

    time_between_prints = 0.5  # 0.5 seconds
    counter = 0
    print('Press Ctrl+C to quit: ')

    # plot a figure showing the ft result for each time
    plt.ion()  # enable interactivity
    fig = plt.figure()  # make a figure

    while stream.is_active():
        time.sleep(time_between_prints)
        # ch = input("Press (p to quit): ")

        left_bandwith = int(Sine_Dop.doppler_shift[0])
        right_bandwith = int(Sine_Dop.doppler_shift[1])
        # # print(doppler_shift)
        # str_left_space = ' ' * (Sine_Dop.relevant_freq_window - left_bandwith)
        # str_left = '<' * left_bandwith
        # str_right = '>' * right_bandwith
        # str_right_space = ' ' * (Sine_Dop.relevant_freq_window - right_bandwith)
        # print(str_left_space, str_left, "||", str_right, str_right_space)
        if left_bandwith > right_bandwith:
            print('                              push backward*************************>')
            print('                              push backward*************************>')
            print('                              push backward*************************>')
            print('                              push backward*************************>')
            print('                              push backward*************************>')
        else:
            print('<*****************************push forward                          ')
            print('<*****************************push forward                          ')
            print('<*****************************push forward                          ')
            print('<*****************************push forward                          ')
            print('<*****************************push forward                          ')

        drawnow(Sine_Dop.makeFig)

        if counter == 100:
            # Times out in 250 cycles.
            # Exit and write to file.
            stream.stop_stream()
        counter += 1
        # print(counter)

    stream.stop_stream()
    stream.close()
    p.terminate()

    # # Save the output to WAV file.
    # WAVE_OUTPUT_FILENAME = "WAV_Doppler_output.wav"
    # # FORMAT = pyaudio.paFloat32
    # FORMAT = pyaudio.paInt16
    #
    # wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    # wf.setnchannels(CHANNELS)
    # wf.setsampwidth(p.get_sample_size(FORMAT))
    # wf.setframerate(SAMPLE_RATE)
    # wf.writeframes(b''.join(frames_to_file))
    # wf.close()
    # print('Session saved to WAV file ... WAV_Doppler_output.wav')
    # print("frame_count_global:", Sine_Dop.frame_count_global)

    # Plot the final 2048 FFT values.
    # fft_data = np.abs(np.fft.fft(Sine_Dop.mic_buffer))
    # print('fft_primary_tone_index: ', Sine_Dop.fft_primary_tone_index)
    # print('fft_data[fft_primary_tone_index]: ', fft_data[Sine_Dop.fft_primary_tone_index])
    #
    # for i in range(820, 850):
    #     print('fft_data[', i, ']: ', fft_data[i])
    #
    # # plt.plot( np.abs( np.fft.fft(mic_buffer) ) )
    # plt.plot(np.abs(np.fft.fft(Sine_Dop.mic_buffer))[820:850])
    # plt.show()
