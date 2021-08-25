import librosa
import soundfile
import numpy as np
import scipy
import seaborn
import matplotlib.pyplot as plt
import sys

def read_sound(filename):
    #data, sr = soundfile.read(filename, dtype='float32')
    data, sr = librosa.load(filename)
    return (data, sr)

def analyze_sound(data, sr, filename,
                full = True,
                pre_max = 10, 
                post_max = 5,
                pre_avg = 50,
                delta = 0.10,
                hop_length = 512):
    # separate into notes
    onset_frames = librosa.onset.onset_detect(data,
                                              sr = sr,
                                              units = "frames",
                                              hop_length = hop_length,
                                              backtrack = True,
                                              pre_max = pre_max,
                                              post_max = post_max,
                                              pre_avg = pre_avg,
                                              delta = delta)

    beats = len_of_notes(onset_frames)
    
    onset_samples = librosa.frames_to_samples(onset_frames, hop_length = hop_length)
    plot_filename = filename.rstrip('.wav')
    plot_chroma(data, sr, onset_samples, beats, plot_filename, full)


def len_of_notes(onset_frames):
    len_of_notes = []
    min_len = onset_frames[-1]
    max_len = 0
    for i in range(1,len(onset_frames)):
        len_of_notes.append(onset_frames[i] - onset_frames[i-1])
        min_len = min(min_len,len_of_notes[-1])
        max_len = max(max_len,len_of_notes[-1])
    beats = []

    for i in range(len(len_of_notes)):
        beats.append(int(round(len_of_notes[i]/min_len,0)))
    return beats

def pickMaxFreqs(freqs, fmin, fmax, spacing, pre_max = 5, threshold = 0.25 ):
    # find all local maxima

    max_index = [0]

    minheight = freqs.max() * threshold
    for i in range(int(fmin/spacing),int(fmax/spacing),1):
        if freqs[i] > freqs[i-1] and freqs[i] > freqs[i+1]:

            
            if i >= max_index[-1] + 5 and freqs[i] >= minheight:
                max_index.append(i)

                
    

    max_freq = [ index * spacing for index in max_index ]
    return max_freq[1:]

def generate_chroma(freqs, n_duration, full = True):
    
    if full:
        #dimension 43, ranging between D2 and G5#, reasonable scale for guitar
        y = np.zeros((43,n_duration))
        for freq in freqs:
            midi = librosa.hz_to_midi(freq)

        # for now limit to lowest D (in D-tuning), expand later. D2 = 38 in midi
            index = int(round(max(0, midi - 38),0))
        
            index = min(38,index) 
        #print(index)
            y[index,:] = 1.
    else:
        y = np.zeros((12, n_duration))
        for freq in freqs:
            midi = librosa.hz_to_midi(freq)
            index = int(round(max(0, midi - 38),0)) % 12

            y[index,:] = 1.

    return y
    
def generate_melody_segment(freqs, sr, n_duration):
    n = np.linspace(0,n_duration, n_duration)
    y = 0.2*np.sin(2*np.pi*0*n/float(sr))
    for freq in freqs:
        y += 0.2*np.sin(2*np.pi*freq*n/float(sr))
    return y

def estimate_pitch(segment,  sr, fmin=50.0, fmax=800.0):
    
    X = scipy.fftpack.fft(segment)
    X_mag = np.absolute(X[0:len(X)//2])
    
    freqs = pickMaxFreqs(X_mag, fmin, fmax, sr/len(X), threshold = 0.25)
    
    if len(freqs) > 6:
        freqs = pickMaxFreqs(X_mag, fmin, fmax, sr/len(X), threshold = 0.50)
    
    return freqs

def plot_chroma(data, sr, onset_samples, beats, filename, full = True):

    if full:
        chroma_fft= np.array([], dtype=np.float32).reshape(43,0)
    else:
        chroma_fft= np.array([], dtype=np.float32).reshape(12,0)
    generated_melody = np.array([0])
    for i in range(len(onset_samples) - 1):  
        start_index = onset_samples[i]
        end_index = onset_samples[i + 1] + 1
        freqs = estimate_pitch(data[start_index:end_index], sr)
        chroma_note = generate_chroma(freqs, beats[i], full = full)
        chroma_fft = np.concatenate((chroma_fft, chroma_note), axis = 1)

        y = generate_melody_segment(freqs, sr, end_index - start_index)
        generated_melody = np.concatenate((generated_melody, y), axis = None)
    
    fig, ax = plt.subplots(figsize=(10,10))
    if full:
        y_ticks = ['D2', 'D2#', 'E2', 'F2', 'F2#', 'G2', 'G2#', 
                'A3', 'A3#', 'B3', 'B3#', 'C3', 'D3', 'D3#', 'E3', 'F3', 'F3#', 'G3', 'G3#', 
                'A4', 'A4#', 'B4', 'B4#', 'C4', 'D4', 'D4#', 'E4', 'F4', 'F4#', 'G4', 'G4#', 
                'A5', 'A5#', 'B5', 'B5#', 'C5', 'D5', 'D5#', 'E5', 'F5', 'F5#', 'G5', 'G5#'] 
    else:
        y_ticks = ['D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B', 'B#', 'C']
    ax = seaborn.heatmap(chroma_fft, yticklabels=y_ticks, xticklabels=False)
    ax.invert_yaxis()
    plt.savefig(filename + '.png')

    soundfile.write(filename + '_generated.wav', generated_melody, sr)

def main():


    if len(sys.argv) < 2:
        print('analyze.py filename')
        exit(1)
    
    
    if '.wav' not in sys.argv[1]:
        print('Please provide a .wav soundfile (short)')
        exit(1)
    filename = sys.argv[1]
    
    full = True
    if len(sys.argv) == 3 and sys.argv[3] == 'short':
        full = False

    (data, sr) = read_sound(filename)
    analyze_sound(data, sr, filename, full)

if __name__ == "__main__":
    main()



    




