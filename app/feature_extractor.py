import librosa
import numpy as np

def extract_features(wav_path):
    y, sr = librosa.load(wav_path, sr = 44100)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr = sr)
    brightness = float(np.mean(spectral_centroid))

    rms = librosa.feature.rms(y=y)
    energy = float(np.mean(rms))

    return {
        "tempo_bpm": float(tempo),
        "brightness": brightness,
        "energy": energy

    }