import numpy as np
import librosa

def extract_audio_features(filename):
    x, sr = librosa.load(filename)

    total_energy = np.sum(x ** 2)
    rmse = np.mean(librosa.feature.rms(y=x))

    zero_crossings = sum(librosa.zero_crossings(x, pad=False))

    tempo = librosa.beat.tempo(y=x)[0]

    mfcc = np.mean(librosa.feature.mfcc(y=x, sr=sr), axis=1)

    delta_mfcc = np.mean(librosa.feature.delta(librosa.feature.mfcc(y=x, sr=sr)), axis=1)

    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=x, sr=sr))

    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=x, sr=sr))

    spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=x, sr=sr), axis=1)

    spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=x))

    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=x, sr=sr))

    freqs = np.fft.fftfreq(x.size)
    mean_freq = np.mean(freqs)
    std_freq = np.std(freqs)
    max_freq = np.amax(freqs)
    min_freq = np.amin(freqs)

    feature_vector = np.array([
        total_energy,
        rmse,
        zero_crossings,
        tempo,
        *mfcc,
        *delta_mfcc,
        spectral_centroid,
        spectral_bandwidth,
        *spectral_contrast,
        spectral_flatness,
        spectral_rolloff,
        mean_freq,
        std_freq,
        max_freq,
        min_freq
    ])

    return feature_vector


prueba = 'c:/Users/Dario/Desktop/BD2/town-10169.mp3'
print(extract_audio_features(prueba))
