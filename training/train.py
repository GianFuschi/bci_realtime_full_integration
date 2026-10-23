import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import yaml

import pickle
from src.features import BandPower
from src.classifier import OnlineLDA 
import numpy as np


class Trainer():
    def __init__(self, config_path="config/config.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        self.cfg = cfg
        
        self.psd = BandPower(fs=cfg["sampling_rate"], band=cfg["features"]["band"])
        self.final_model = OnlineLDA()
        self.cv_pipeline = Pipeline([('classifier', LinearDiscriminantAnalysis())])
        self.folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def load_epochs(self):
        file_path = self.cfg["features"]["raw_epochs"]
        with open(file_path, "rb") as f:
            epochs, labels = pickle.load(f)
        return epochs, labels

    def epoch_to_features(self, epochs):
        """
        Convert epochs to features using band power extraction.
        Note that the data is already filtered using lfilter. 
        """
        features = []
        channel_idx = self.cfg["features"]["channel_indeces"]

        for epoch in epochs:

            epoch = epoch[:, channel_idx] #assuming epoch shape is (samples, channels)
            feat = self.psd.compute(epoch)
            print("WE CAN EXPERIMENT HERE WITH AND WITHOUT LOGGING -> supposed to help with normal dist assumption LDA")
            feat = np.log10(feat)  # Log-transform the features
            features.append(feat)
    
        return np.array(features)
        
    def train_lda(self, features, labels):
        """
        Train an LDA classifier using the provided features and labels.
        """
        scores = cross_val_score(self.cv_pipeline,
                                 features,
                                 labels,
                                 cv=self.folds,
                                 scoring='balanced_accuracy',
                                 error_score='raise')
        
        print(f"Cross-validated accuracy: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
        print(f"Individual fold accuracies: {scores}")
        
        self.final_model.train(features, labels)
        self.final_model.save(self.cfg["classifier"]["path"])

    def save_features(self, features, labels, label):
        f = features[labels == label]
        f = pd.DataFrame(f)
        f.to_csv(f"recorded_features_{label}.csv", index=False, header=False)

    def train(self):
        epochs, labels = self.load_epochs()
        features = self.epoch_to_features(epochs)
        unique_labels = np.unique(labels)
        for label in unique_labels:
            self.save_features(features, labels, label)
        self.train_lda(features, labels)
        
    
trainer = Trainer()
trainer.train()
