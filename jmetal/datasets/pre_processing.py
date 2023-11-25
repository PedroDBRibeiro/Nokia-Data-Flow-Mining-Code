from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
from sklearn import preprocessing

import pandas as pd
import numpy as np

seed = 1603
np.random.seed(seed)

class Data:
    def __init__(self, filename):
        self.data = pd.read_csv(filename, index_col = 0)

    def getY(self):
        group = self.data['Group'].values
        le = preprocessing.LabelEncoder()
        y=le.fit_transform(group)
        unique, counts = np.unique(group, return_counts=True)
        print(dict(zip(unique, counts)))
        return y
    def getX(self):
        x = self.data.drop('Group', axis=1).values
        return x

    def normalizeX(self):
        normalizer = preprocessing.Normalizer().fit(self.getX())
        normX = normalizer.transform(self.getX())
        return(normX)


    def norm_DataSets(self):
        print("Getting Original Data")
        X = self.normalizeX()
        #print(X)
        y = self.getY()
        X_Pretrain, X_test, y_Pretrain, y_test = train_test_split(X, y, test_size=.25, random_state=seed,
                                                                      stratify=y)

        X_train, X_validation, y_train, y_validation = train_test_split(X_Pretrain, y_Pretrain, test_size = .25,
                                                                        random_state=seed, stratify=y_Pretrain)

        return({'X_train':X_train,'y_train':y_train,'X_test':X_test,'y_test':y_test, 'X_validation':X_validation,
                'y_validation':y_validation})


    def original_DataSets(self):
        print("Getting Original Data")
        X = self.getX()
        y = self.getY()
        X_Pretrain, X_test, y_Pretrain, y_test = train_test_split(X, y, test_size=.25, random_state=seed,
                                                                      stratify=y)

        X_train, X_validation, y_train, y_validation = train_test_split(X_Pretrain, y_Pretrain, test_size = .25,
                                                                        random_state=seed, stratify=y_Pretrain)

        return({'X_train':X_train,'y_train':y_train,'X_test':X_test,'y_test':y_test, 'X_validation':X_validation,
                'y_validation':y_validation})
    def ADASYN_DataSets(self):
        print("Oversampling with ADASYN")
        X = self.getX()
        y = self.getY()
        X_Pretrain,X_validation , y_Pretrain, y_validation = train_test_split(X, y, test_size=.25, random_state=seed,
                                                                      stratify=y)
        ada = ADASYN(sampling_strategy='minority', random_state=seed)
        X_resampled, y_resampled  = ada.fit_resample(X_Pretrain, y_Pretrain)
        X_train,X_test , y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=.25,
                                                                        random_state=seed, stratify=y_resampled)
        return({'X_train':X_train,'y_train':y_train,'X_test':X_test,'y_test':y_test, 'X_validation':X_validation,
                'y_validation':y_validation})

    def ROS_DataSets(self):
        print("ROS data")
        X = self.getX()
        y = self.getY()
        X_Pretrain, X_validation, y_Pretrain, y_validation = train_test_split(X, y, test_size=.25, random_state=seed,
                                                                  stratify=y)
        rndO = RandomOverSampler(sampling_strategy="all", random_state=seed)
        X_resampled, y_resampled = rndO.fit_resample(X_Pretrain, y_Pretrain)
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=.25,
                                                                        random_state=seed,
                                                                        stratify=y_resampled)
        return (
        {'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test, 'X_validation': X_validation,
         'y_validation': y_validation})

    def RUS_DataSets(self):
        print("RUS data")
        X = self.getX()
        y = self.getY()
        X_Pretrain,X_validation, y_Pretrain,y_validation  = train_test_split(X, y, test_size=.25, random_state=seed,
                                                                  stratify=y)
        rndU = RandomUnderSampler(sampling_strategy="majority", random_state=seed)
        X_resampled, y_resampled = rndU.fit_resample(X_Pretrain, y_Pretrain)
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=.25,
                                                                        random_state=seed,
                                                                        stratify=y_resampled)
        return (
        {'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test, 'X_validation': X_validation,
         'y_validation': y_validation})

    def SMOTE_DataSets(self):
        print("Oversampling with SMOTE")
        X = self.getX()
        y = self.getY()
        X_Pretrain, X_validation, y_Pretrain, y_validation = train_test_split(X, y, test_size=.25, random_state=seed,
                                                                              stratify=y)
        smt = SMOTE(sampling_strategy='minority', random_state = seed)
        X_resampled, y_resampled = smt.fit_resample(X_Pretrain, y_Pretrain)
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=.25,
                                                                        random_state=seed,stratify=y_resampled)

        return({'X_train':X_train,'y_train':y_train,'X_test':X_test,'y_test':y_test, 'X_validation':X_validation,
                'y_validation':y_validation})

    def SMOTETomek_DataSets(self):
        print("Oversampling with SMOTE-Tomek")
        X = self.getX()
        y = self.getY()
        X_Pretrain, X_validation, y_Pretrain, y_validation = train_test_split(X, y, test_size=.25, random_state=seed,
                                                                              stratify=y)
        smttmk = SMOTETomek(sampling_strategy='minority', random_state=seed)
        X_resampled, y_resampled = smttmk.fit_resample(X_Pretrain, y_Pretrain)
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=.25,
                                                                        random_state=seed,
                                                                        stratify=y_resampled)

        return({'X_train':X_train,'y_train':y_train,'X_test':X_test,'y_test':y_test, 'X_validation':X_validation,
                'y_validation':y_validation})