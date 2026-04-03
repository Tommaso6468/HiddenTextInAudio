import numpy as np
import wavio
import matplotlib.pyplot as plt

def add_band(signal, startTs, endTs, startHz, endHz, rate, fade_ms=50):
    startTs = int(startTs * rate)
    endTs = int(endTs * rate)
    
    length = endTs - startTs
    if length <= 0:
        return
    
    noise = np.random.randn(length)
    fft_noise = np.fft.rfft(noise)
    freqs = np.fft.rfftfreq(length, 1/rate)
    mask = (freqs >= startHz) & (freqs <= endHz)
    fft_noise[~mask] = 0
    filtered = np.fft.irfft(fft_noise, n=length)
    
    max_val = np.max(np.abs(filtered))
    if max_val < 1e-10:
        return
    filtered /= max_val
  
    fade_samples = int(fade_ms * rate / 1000)
    fade_samples = min(fade_samples, length // 2)
    fade_in = np.hanning(fade_samples * 2)[:fade_samples]
    fade_out = np.hanning(fade_samples * 2)[fade_samples:]
    envelope = np.ones(length)
    envelope[:fade_samples] = fade_in
    envelope[-fade_samples:] = fade_out
    filtered *= envelope
    
    signal[startTs:endTs] += filtered
    
def add_diagonal_band(signal, startTs, endTs, startHz, endHz, targetHz_start, targetHz_end, rate, fade_ms=50):
    num_slices = 10
    slice_times = np.linspace(startTs, endTs, num_slices + 1)
    
    slice_startHz = np.linspace(startHz, targetHz_start, num_slices)
    slice_endHz = np.linspace(endHz, targetHz_end, num_slices)
    
    for i in range(num_slices):
        add_band(signal, slice_times[i], slice_times[i+1], slice_startHz[i], slice_endHz[i], rate, fade_ms=fade_ms)
     
# bar width vertical =  0.05s
# bar width horizontal full = 0.7s 
# bar height horizontal = 10Hz
# segment bottom = 200Hz, top = 2000 Hz
# startHz, endHz, startTs in display, endTs in display
CHAR_WIDTH = 0.7
SEG_HZ = {
    'a': (1990, 2000, 0.05, 0.65),
    'b': (1200, 1900, 0.65, 0.7),
    'c': (300, 1000, 0.65, 0.7),
    'd': (200, 210, 0.05, 0.65),
    'e': (300, 1000, 0, 0.05),
    'f': (1200, 1900, 0, 0.05),
    'g1': (1095, 1105, 0.05, 0.325),
    'g2': (1095, 1105, 0.375, 0.65),
    'h': (1200, 1900, 0.325, 0.375),
    'i': (300, 1000, 0.325, 0.375),
    'n': (50, 60, 0, 0.05)
}

DIAG_SEG_HZ = {
    'j': (1990, 2000, 1095, 1105, 0, 0.35),
    'k': (1095, 1105, 1990, 2000, 0.35, 0.7),
    'm': (200, 210, 1095, 1105, 0, 0.35),
    'l': (1095, 1105, 200, 210, 0.35, 0.7),
    'o': (50, 60, 200, 210, 0, 0.2)
}

def display_segment(seg, startTs, signal, rate):
    if seg in SEG_HZ:
        startHz, endHz, segmentStartTs, segmentEndTs = SEG_HZ[seg]
        add_band(
            signal,
            startTs + segmentStartTs,
            startTs + segmentEndTs,
            startHz,
            endHz,
            rate
        )
    elif seg in DIAG_SEG_HZ:
        startHz, endHz, targetHz_start, targetHz_end, segmentStartTs, segmentEndTs = DIAG_SEG_HZ[seg]
        add_diagonal_band(
            signal,
            startTs + segmentStartTs,
            startTs + segmentEndTs,
            startHz,
            endHz,
            targetHz_start,
            targetHz_end,
            rate
        )
    
#        a
#  -------------
#  |\   h|    /|
# f| \j  |  k/ |b
#  |  \  |  /  |
#  |   \ | /   |
#g1------ ------g2 
#  |   / | \   |
#  |  /  |  \  |
# e| /m  |  l\ |c
#  |/   i|    \|
#  -------------
#  .,    d
#  no
    
CHARACTERS = {
    'A': ('f', 'b', 'g1', 'g2', 'e', 'c', 'a'),
    'B': ('a', 'b', 'c', 'd', 'g2', 'h', 'i'),
    'C': ('a', 'd', 'e', 'f'),
    'D': ('a', 'b', 'c', 'd', 'h', 'i'),
    'E': ('a', 'd', 'e', 'f', 'g1', 'g2'),
    'F': ('a', 'e', 'f', 'g1', 'g2'),
    'G': ('a', 'c', 'd', 'e', 'f', 'g2'),
    'H': ('b', 'c', 'e', 'f', 'g1', 'g2'),
    'I': ('a', 'd', 'h', 'i'),
    'J': ('b', 'c', 'd', 'e'),
    'K': ('e', 'f', 'g1', 'k', 'l'),
    'L': ('d', 'e', 'f'),
    'M': ('b', 'c', 'e', 'f', 'j', 'k'),
    'N': ('b', 'c', 'e', 'f', 'j', 'l'),
    'O': ('a', 'b', 'c', 'd', 'e', 'f'),
    'P': ('a', 'b', 'e', 'f', 'g1', 'g2'),
    'Q': ('a', 'b', 'c', 'd', 'e', 'f', 'l'),
    'R': ('a', 'b', 'e', 'f', 'g1', 'g2', 'l'),
    'S': ('a', 'c', 'd', 'f', 'g1', 'g2'),
    'T': ('a', 'h', 'i'),
    'U': ('b', 'c', 'd', 'e', 'f'),
    'V': ('e', 'f', 'm', 'k'),
    'W': ('b', 'c', 'e', 'f', 'm', 'l'),
    'X': ('j', 'k', 'm', 'l'),
    'Y': ('i', 'j', 'k'),
    'Z': ('a', 'd', 'm', 'k'),
    '1': ('b', 'c', 'k'),
    '2': ('a', 'b', 'g1', 'g2', 'e', 'd'),
    '3': ('a', 'b', 'g1', 'g2', 'c', 'd'),
    '4': ('f', 'g1', 'g2', 'b', 'c'),
    '5': ('a', 'f', 'g1', 'g2', 'c', 'd'),
    '6': ('a', 'f', 'g1', 'g2', 'e', 'c', 'd'),
    '7': ('a', 'b', 'c'),
    '8': ('a', 'b', 'c', 'd', 'e', 'f', 'g1', 'g2'),
    '9': ('a', 'b', 'c', 'd', 'f', 'g1', 'g2'),
    '0': ('a', 'b', 'c', 'd', 'e', 'f', 'm', 'k'),
    '+': ('g1', 'g2', 'h', 'i'),
    '-': ('g1', 'g2'),
    '/': ('m', 'k'),
    '\\': ('j', 'l'),
    '?': ('a', 'b', 'f', 'g2', 'i'),
    '*': ('g1', 'g2', 'h', 'i', 'j', 'k', 'l', 'm'),
    '(': ('k', 'l'),
    ')': ('j', 'm'),
    '.': ('n'),
    '!': ('n', 'e', 'f'),
    '=': ('g1', 'g2', 'd'),
    ',': ('o')
}

def display_char(char, startTs, signal, rate):
    char = char.upper()
    segments = CHARACTERS[char]
    for seg in segments:
        display_segment(seg, startTs, signal, rate)
   
WORD_START_TS = 0.1
WORD_GAP_TS = 0.3
CHAR_GAP_TS = 0.1

def display_text(text, signal, rate):
    words = text.split()
    currentTs = WORD_START_TS
    for word in words:
        for char in word:
            display_char(char, currentTs, signal, rate)
            currentTs += CHAR_WIDTH + CHAR_GAP_TS
        currentTs += WORD_GAP_TS - CHAR_GAP_TS
        
def audio_length(text):
    return WORD_START_TS + len(text) * (CHAR_WIDTH + CHAR_GAP_TS) - CHAR_GAP_TS + text.count(' ') * (WORD_GAP_TS - CHAR_GAP_TS - CHAR_WIDTH) + 0.1

print("Enter the desired text")
text = input()  

rate = 22050
T = audio_length(text)
n = int(rate*T)
t = np.arange(n) / rate
signal = np.zeros(n)

display_text(text, signal, rate)

signal /= np.max(np.abs(signal))

wavio.write("audio.wav", signal, rate, sampwidth=3)

wav = wavio.read("audio.wav")
rate = wav.rate
data = wav.data.squeeze()

plt.figure(figsize=(len(text), 4))
with np.errstate(divide='ignore'):
    plt.specgram(data, Fs=rate, NFFT=1024, noverlap=512, cmap="magma")
plt.ylim(0, 5000)
plt.show()

