# melody_analyzer
Analyzes the frequencies and rhythm of a .wav file and reproduces a computer-made melody

Usage: python3 analyze.py soundfile.wav

The program takes a sound file of .wav format as input and outputs a chromatogram and generated melody.

The melody is first separated into distinct notes using an onset envelope with the help of the librosa package. The parameters are optimized for a guitar but can also be applied to other instruments. The rhythm of the melody is obtained from the relative lengths of each note. Then, the main frequencies are calculated for each note using FFT with the help of the scipy package. The program generates two files:
1) A chromatogram showing the main frequencies for each note.
2) A computer-generated melody from the main frequencies.

Several examples can be found in the examples folder.
