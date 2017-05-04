#This is an exampe of a genetic algorithm using the TPOT dependency
from tpot import TPOT
from.sklearn.cross_validation import train_test_split
import pandas as pd
import numpy as np

#Data load
eMagData = pd.read_csv('GammaRayScope.csv')

#cleaning
eMagData_Mix = eMagData.iloc[np.radom.permutation(len(eMagData))]
eMagClean = eMagDataMix.reset_index(drop=True)

#classifying
eMagClean['Class'] = eMagClean['Class'].map({'g' : 0, 'h' : 1})
eMagClass = eMagClean['Class'].values

#training
training_indices, validation_indices = training_indices, testing_indices, = train_test_split(eMagClean.index, stratify = eMagClass, train_size= 0.66, test_size = 0.33)

#genetic addition using TPOT
tpot = TPOT(generations = 13, verbosity = 2)
tpot.fit(eMagClean.drop('Class', axis = 1).loc[training_indices].values, tele.loc[training_indices, 'Class'].values)

#Scoring
score = tpot.score(eMagClean.drop('Class', axis=1).loc[validation_indices].values, eMagClean.loc[validation_indices, 'Class'].values)
