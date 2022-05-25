# Sound-based Gesture Sensing by Analyzing Doppler
This is an inofficial implementation of paper ['SoundWave: Using the Doppler Effect to Sense Gestures'](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/11/GuptaSoundWaveCHI2012.pdf) running on laptop or PC. The code is adapted from github repository from [Joao Nuno Carvalho](https://github.com/joaocarvalhoopen/computer_Doppler_RADAR)

This project leverages the speaker and microphone already embedded in most commodity devices to sense in-air gestures around the device. To do this, we generate an inaudible tone, which gets frequency-shifted when it reflects off moving objects like the hand. We measure this shift with the microphone to infer various gestures.

To test this program, you should disable in the microphone settings, the echo cancelation and the noise reduction. Control Panel -> Sound -> Recording -> Microphone Array -> Propertoes -> Advanced -> Disable Audio Enhancements

Demo of running code in real time:
![Watch Video](./files/hw3_demo.mp4)