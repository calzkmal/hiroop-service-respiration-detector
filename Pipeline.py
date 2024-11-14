import librosa
import numpy as np

EXPECTED_FEATURE_LENGTH = 162  # Panjang yang diharapkan oleh model

def extract_features(filepath):
    try:
        # Load audio file
        data, sample_rate = librosa.load(filepath, duration=2.5, offset=0.6)
        
        # Zero Crossing Rate
        zcr = np.mean(librosa.feature.zero_crossing_rate(data).T, axis=0)
        
        # MFCC
        mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=13).T, axis=0)
        
        # Chroma Feature
        chroma = np.mean(librosa.feature.chroma_stft(y=data, sr=sample_rate).T, axis=0)
        
        # Mel Spectrogram
        mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
        
        # Combine all features into a single array
        features = np.hstack([zcr, mfcc, chroma, mel])
        
        # Pad or crop features to match the expected length
        if len(features) < EXPECTED_FEATURE_LENGTH:
            features = np.pad(features, (0, EXPECTED_FEATURE_LENGTH - len(features)), 'constant')
        else:
            features = features[:EXPECTED_FEATURE_LENGTH]
        
        return features
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None
