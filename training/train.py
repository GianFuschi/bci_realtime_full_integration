import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import yaml

import pickle
from src.features import BandPower

CHANNEL_IDX = [1, 3] 

CONFIG_PATH = "config/config.yaml"
with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)

def load_epochs(file_path):
    with open(file_path, "rb") as f:
        epochs, labels = pickle.load(f)
    return epochs, labels

def epoch_to_features(epochs, fs=CFG["sampling_rate"], band=CFG["features"]["band"]):
    """
    Convert epochs to features using band power extraction.
    Note that the data is already filtered using lfilter. 
    """
    psd = BandPower(fs = fs, band = band)
    features = []
    for epoch in epochs:
        epoch = epoch[:, CHANNEL_IDX] #assuming epoch shape is (samples, channels)
        feat = psd.compute(epoch)
        feat = np.log10(feat)  # Log-transform the features
        features.append(feat)
    return np.array(features)
    
def train_lda(features, labels):
    """
    Train an LDA classifier using the provided features and labels.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pipeline = Pipeline([('scaler', StandardScaler()),
                          ('classifier', LinearDiscriminantAnalysis())])

    print(labels)
    print(features)

    scores = cross_val_score(pipeline, features, labels, cv=cv, scoring='balanced_accuracy', error_score='raise')
    # cross_val_score only returns the scores, not the fitted model. We have to fit it to save it.
    pipeline.fit(features, labels)
        
    print(f"Cross-validated accuracy: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    print(f"Individual fold accuracies: {scores}")

    return pipeline

epochs, labels = load_epochs("src/epochs.pkl")
features = epoch_to_features(epochs)
print(features.shape)

features_left = features[labels == 'IMAGERY_LEFT']
print(len(features_left))
features_right = features[labels == 'IMAGERY_RIGHT']

data_left = pd.DataFrame(features_left)
data_left.to_csv("recorded_features_left.csv", index=False, header=False)
data_right = pd.DataFrame(features_right)
data_right.to_csv("recorded_features_right.csv", index=False, header=False)
pipeline = train_lda(features, labels)
with open('model_weights.pkl', 'wb') as f:
        pickle.dump(pipeline, f)