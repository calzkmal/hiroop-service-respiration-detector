import librosa
import numpy as np

class FeatureExtractor:
    EXPECTED_FEATURE_LENGTH = 162  # Length expected by the model

    def __init__(self, duration=2.5, offset=0.6):
        self.duration = duration
        self.offset = offset

    def extract_features(self, data):
        # Zero Crossing Rate
        zcr = np.mean(librosa.feature.zero_crossing_rate(data).T, axis=0)
        
        # Chroma_stft
        stft = np.abs(librosa.stft(data))
        chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft).T, axis=0)
        
        # MFCC
        mfcc = np.mean(librosa.feature.mfcc(y=data).T, axis=0)
        
        # Root Mean Square Value
        rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
        
        # MelSpectogram
        mel = np.mean(librosa.feature.melspectrogram(y=data).T, axis=0)
        
        # Combine all features into a single array
        features = np.hstack([zcr, chroma_stft, mfcc, rms, mel])
        
        # Pad or crop features to match the expected length
        if len(features) < self.EXPECTED_FEATURE_LENGTH:
            features = np.pad(features, (0, self.EXPECTED_FEATURE_LENGTH - len(features)), 'constant')
        else:
            features = features[:self.EXPECTED_FEATURE_LENGTH]
        
        return features

    def add_noise(self, data):
        noise_amp = 0.035 * np.random.uniform() * np.amax(data)
        return data + noise_amp * np.random.normal(size=data.shape[0])

    def stretch(self, data, rate=0.8):
        return librosa.effects.time_stretch(data, rate=rate)

    def shift(self, data):
        shift_range = int(np.random.uniform(low=-5, high=5) * 1000)
        return np.roll(data, shift_range)

    def pitch(self, data, sampling_rate, pitch_factor=0.7):
        return librosa.effects.pitch_shift(data, sr=sampling_rate, n_steps=pitch_factor)

    def get_features(self, path):
        # Load audio file
        data, sample_rate = librosa.load(path, duration=self.duration, offset=self.offset)
        
        # Extract features without augmentation
        return self.extract_features(data)

    def get_aug_features(self, path):
        # Load audio file
        data, sample_rate = librosa.load(path, duration=self.duration, offset=self.offset)
        
        # Original features
        features = self.extract_features(data)
        result = np.array([features])
        
        # Augmented features
        noise_data = self.add_noise(data)
        noise_features = self.extract_features(noise_data)
        result = np.vstack((result, noise_features))
        
        stretch_data = self.stretch(data)
        stretch_pitch_data = self.pitch(stretch_data, sample_rate)
        stretch_pitch_features = self.extract_features(stretch_pitch_data)
        result = np.vstack((result, stretch_pitch_features))
        
        return result