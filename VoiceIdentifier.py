!pip install librosa
import os
import zipfile
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers, models
import librosa.display
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.decomposition import PCA
from tensorflow.keras.callbacks import EarlyStopping
from pydub import AudioSegment
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers import Adam
def convert_to_wav(file_path, output_dir):
    """Converts an audio file to WAV format."""
    # Determine file extension
    ext = os.path.splitext(file_path)[-1].lower()
    try:
        if ext in ['.m4a', '.mp3', '.ogg']:
            audio = AudioSegment.from_file(file_path)
            output_path = os.path.join(output_dir, os.path.splitext(os.path.basename(file_path))[0] + ".wav")
            audio.export(output_path, format="wav")
            return output_path
        elif ext == '.wav':
            return file_path
        else:
            print(f"Unsupported file format: {file_path}")
            return None
    except Exception as e:
        print(f"Error converting {file_path}: {e}")
        return None

# Create a folder for converted .wav files
converted_dir = "/content/converted_audio"
os.makedirs(converted_dir, exist_ok=True)

# Convert all audio files in the dataset
for speaker in os.listdir(data_dir):
    speaker_folder = os.path.join(data_dir, speaker)
    if os.path.isdir(speaker_folder):
        for file in os.listdir(speaker_folder):
            file_path = os.path.join(speaker_folder, file)
            converted_file = convert_to_wav(file_path, converted_dir)
            if converted_file:
                print(f"Converted: {file_path} -> {converted_file}")

def plot_waveform(file_path):
    signal, sample_rate = librosa.load(file_path, sr=None)
    plt.figure(figsize=(10, 4))
    plt.plot(signal)
    plt.title("Time-Domain Waveform")
    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude")
    plt.show()
    return signal, sample_rate
def plot_frequency_domain(signal, sample_rate):
    fft_values = np.fft.fft(signal)
    fft_magnitude = np.abs(fft_values)
    frequencies = np.fft.fftfreq(len(fft_values), d=1/sample_rate)

    plt.figure(figsize=(10, 4))
    plt.plot(frequencies[:len(frequencies)//2], fft_magnitude[:len(fft_magnitude)//2])
    plt.title("Frequency-Domain Representation")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.show()
    def plot_spectrogram(signal, sample_rate):
    stft = librosa.stft(signal)
    spectrogram = np.abs(stft)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.amplitude_to_db(spectrogram, ref=np.max),
                             sr=sample_rate, x_axis='time', y_axis='log')
    plt.title("Spectrogram (Log-Frequency Scale)")
    plt.colorbar(format="%+2.0f dB")
    plt.xlabel("Time")
    plt.ylabel("Frequency (Hz)")
    plt.show()
    def extract_fourier_features(signal, sample_rate, n_features=500):
    fft_values = np.fft.fft(signal)
    fft_magnitude = np.abs(fft_values)[:n_features]
    return fft_magnitude
# Initialize arrays to hold features and labels
X = []
y = []# Function for data augmentation
def augment_audio(audio_file):
    y, sr = librosa.load(audio_file, sr=None)
    augmented_data = []

    # Add noise
    noise = np.random.randn(len(y)) * 0.005
    y_noisy = y + noise
    augmented_data.append((y_noisy, sr))

    # Time stretch
    y_stretched = librosa.effects.time_stretch(y, rate=1.1)
    augmented_data.append((y_stretched, sr))

    # Pitch shift
    y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=2)
    augmented_data.append((y_shifted, sr))

    return augmented_data

# Function to process audio from data (instead of file path)
def process_audio_from_data(y, sr):
    # Fourier Transform (FFT) to get magnitude
    fft = np.fft.fft(y)
    magnitude = np.abs(fft)[:len(fft)//2]

    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    # Combine both features into one vector
    combined_features = np.concatenate([magnitude, mfccs.flatten()])
    return combined_features

# Process files and apply augmentation
X, y = [], []
for speaker in os.listdir(data_dir):
    speaker_folder = os.path.join(data_dir, speaker)
    if os.path.isdir(speaker_folder):
        for file in os.listdir(speaker_folder):
            if file.endswith((".wav", ".m4a", ".mp3", ".ogg")):
                file_path = os.path.join(speaker_folder, file)

                # Load the audio file
                y_audio, sr = librosa.load(file_path, sr=None)

                # Extract features for the original file
                combined_features = process_audio_from_data(y_audio, sr)
                X.append(combined_features)
                y.append(speaker)

                # Augment and extract features for augmented data
                augmented_audios = augment_audio(file_path)
                for augmented_audio, sr in augmented_audios:
                    # Process augmented audio directly
                    augmented_combined_features = process_audio_from_data(augmented_audio, sr)
                    X.append(augmented_combined_features)
                    y.append(speaker)

print(f"Processed {len(X)} audio files (including augmentations).")

# Pad the feature array so all have the same length
X_padded = pad_sequences(X, padding='post', dtype='float32')

# Encode speaker names to numeric labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Convert features and labels to numpy arrays
X = np.array(X_padded)
y = np.array(y_encoded)

print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reduce dimensionality using PCA
n_components = 13
pca = PCA(n_components=n_components)

# Apply PCA to reduce the feature size of X
X_reduced = pca.fit_transform(X)

print("Shape of X after PCA:", X_reduced.shape)

# Train/test split again with the reduced features
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y_encoded, test_size=0.2, random_state=42)

# Normalize the data again after PCA
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the model
model = models.Sequential([
    layers.InputLayer(input_shape=(X_train.shape[1],)),  # Input size is the reduced number of features
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(np.unique(y_encoded)), activation='softmax')  # Output size matches the number of speakers
])

# Compile the model
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc}")


if not os.path.exists('models'):
    os.makedirs('models')

model.save('models/speaker_recognition.h5')
if os.path.exists('models/speaker_recognition.h5'):
    print("Model has been saved successfully!")
else:
    print("Model was not saved.")
    from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('models/speaker_recognition.h5')

# Class names (adjust according to your dataset)
class_names = ['Sven', 'Riyad', 'Aditiya', 'Rayyan']

# Prediction function for an audio file
def predict_speaker(file_path):
    # Load the audio file
    signal, sample_rate = librosa.load(file_path, sr=None)

    # Extract features from the audio
    combined_features = extract_fourier_features(signal, sample_rate, n_features=13)
    combined_features = np.reshape(combined_features, (1, -1))

    # Make prediction
    prediction = model.predict(combined_features)
    predicted_class = np.argmax(prediction, axis=1)

    # Get the class name based on predicted class index
    predicted_class_name = class_names[predicted_class[0]]

    # Return the predicted class name and prediction probabilities
    return predicted_class_name, prediction

# Main function for prediction and visualization
def predict_and_visualize(file_path):
    # Make the prediction
    predicted_class_name, prediction = predict_speaker(file_path)

    # Visualizations
    print(f"Predicted Speaker: {predicted_class_name}")
    print(f"Prediction Probabilities: {prediction}")

    # Visualize waveform
    plot_waveform(file_path)

    # Visualize frequency domain
    signal, sample_rate = librosa.load(file_path, sr=None)
    plot_frequency_domain(signal, sample_rate)

    # Visualize spectrogram
    plot_spectrogram(signal, sample_rate)

# File path of the test audio
file_path = 'test.wav'
predict_and_visualize(file_path)
