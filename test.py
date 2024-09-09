import librosa
import numpy as np
import joblib

def runtest(example_file_path):
    try:
        loaded_model = joblib.load("models/random_forest_model.joblib")
    except Exception as e:
        return f"Error loading model: {e}"

    def extract_features(file_path):
        try:
            audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
            mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
            return mfccs
        except Exception as e:
            print(f"Error encountered while parsing file {file_path}: {e}")
            return None

    example_features = extract_features(example_file_path)
    if example_features is not None:
        try:
            prediction = loaded_model.predict([example_features])
            class_label = "Real" if prediction[0] == 1 else "Fake"
            return f"{class_label} Audio File"
        except Exception as e:
            return f"Error making prediction: {e}"
    else:
        return "Error extracting features from the example file."
