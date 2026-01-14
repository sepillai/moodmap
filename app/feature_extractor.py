import librosa
import numpy as np

def extract_features(wav_path):
    """
    Extract comprehensive audio features for ML training and analysis.
    Returns both basic features (for API) and extended features (for ML).
    """
    y, sr = librosa.load(wav_path, sr=44100)
    
    # Basic features (for API response)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    brightness = float(np.mean(spectral_centroid))
    rms = librosa.feature.rms(y=y)
    energy = float(np.mean(rms))
    
    # Extended features (for ML training)
    # Spectral features
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    
    # Chroma features (harmonic content)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = float(np.mean(chroma))
    
    # MFCC (Mel-frequency cepstral coefficients) - 13 coefficients
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = [float(np.mean(mfcc)) for mfcc in mfccs]
    
    # Tonnetz (harmonic network)
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    tonnetz_mean = [float(np.mean(t)) for t in tonnetz]
    
    # Rhythm features
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    onset_rate = len(onset_frames) / (len(y) / sr)  # onsets per second
    
    # Harmonic and percussive separation
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    harmonic_ratio = float(np.mean(np.abs(y_harmonic)) / (np.mean(np.abs(y)) + 1e-10))
    
    # Dynamic range
    dynamic_range = float(np.max(y) - np.min(y))
    
    # Return both basic (for API) and extended (for ML)
    return {
        # Basic features (backward compatible)
        "tempo_bpm": float(tempo),
        "brightness": brightness,
        "energy": energy,
        
        # Extended features (for ML)
        "spectral_rolloff": float(np.mean(spectral_rolloff)),
        "spectral_bandwidth": float(np.mean(spectral_bandwidth)),
        "zero_crossing_rate": float(np.mean(zero_crossing_rate)),
        "chroma_mean": chroma_mean,
        "mfcc_means": mfcc_mean,  # List of 13 values
        "tonnetz_means": tonnetz_mean,  # List of 6 values
        "onset_rate": onset_rate,
        "harmonic_ratio": harmonic_ratio,
        "dynamic_range": dynamic_range,
        "duration": len(y) / sr,  # Duration in seconds
    }