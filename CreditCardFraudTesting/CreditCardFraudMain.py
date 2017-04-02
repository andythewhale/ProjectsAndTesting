#This is the main file for the credit card fraud data.
#This file is meant to be compiled in an iPython notebook (Jupyter Notebook)
#Let us do as the romans, and import our main libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline

#Lets get the data and take a look.
ccf = pd.read_csv("C:/Users/../DataSets/creditcardfraud.csv")
ccf.head(10)

#Looking at the instances of fraud vs the instances of non-fraud
#0 is non-fraud, 1 is fraud.
graphData = ccf['Class'].value_counts().sort_index()
graphData.plot(kind='bar')
plt.title=('Instances of Fraud vs Non-Fraud')
plt.xlabel('Class 1 = Fraud, 0 = NonFraud')
plt.ylabel('Number of instances')

#So we can't see anything so lets just ask with code:
print(ccf['Class'].value_counts())

#Amount needs to be normalized, lets check out the bigged transaction first.
#Let's also find the smallest one and the average.
#This is good to know
print("The largest transaction is $",ccf['Amount'].max(), ", nice, a car maybe?")
print("The smallest transaction is $",ccf['Amount'].min(), ", \"yes, i'd like to buy one nothing please.\"")
print("The largest transaction is $ %.2f" % ccf['Amount'].mean(), ", ok, so like a pair of Nike running shoes.")

#So we drop time because it's pretty useless, and we drop the non-normalized amount
#I'm going to keep the original data set in memory, but sometimes this isn't a
#good idea. change ccfN to ccf if you want to be more efficient.
from sklearn.preprocessing import StandardScaler
ccf['nAmount'] = StandardScaler().fit_transform(ccf['Amount'].reshape(-1,1))
ccfN=ccf.drop(['Time', 'Amount'],axis=1)
ccfN.head(10)

print("The largest normalized transaction is $ %.2f" % underSampled_ccfN['nAmount'].max(), ", This is the new car")
print("The smallest normalized transaction is $ %.2f" %ccfN['nAmount'].min(), ", This is the new 0")
print("The average normalized transaction is $ %.2f" % ccfN['nAmount'].mean(), ", This is the new running shoe")

#Samples in the tiny class
fraudCases = len(ccfN[ccfN.Class==1]) #fraud = 492 pieces
fraudIndices = np.array(ccfN[ccfN.Class==1].index) #tells us the row in array format. For Fraud

nonFraud = len(ccfN[ccfN.Class==0]) #non-fraud = 284,315 pieces
nonFraudIndices = np.array(ccfN[ccfN.Class==0].index) #tells us the row in array format. For non-fraud

#Great but we need an undersampled version of the non-Fraud indices
randNonFraudIndices = np.random.choice(nonFraudIndices, fraudCases, replace=False) #Generates 492 random non-fraud case indices for us.

#Combining and pulling data
UnderSampledIndices = np.concatenate([randNonFraudIndices, fraudIndices]) #Combines the indices.
underSampled_ccfN = ccfN.iloc[UnderSampledIndices,:] #Grabs us the data that our undersampled indices points to

#For training and testing large data:
x=ccfN.ix[:,ccfN.columns != 'Class']
y=ccfN.ix[:,ccfN.columns == 'Class']
#Creates 2 data sets, one with class and indices, (y), and one with attributes and no class, (x)

#Testing new data:
xU=underSampled_ccfN.ix[:,underSampled_ccfN.columns != 'Class'] #x from above just undersampled
yU=underSampled_ccfN.ix[:,underSampled_ccfN.columns == 'Class'] #y from above just undersampled

#Checking new dataset, compare this with the old data.
print("Percent Non-Fraud in Undersampled Data: ", len(randNonFraudIndices)/len(underSampled_ccfN))
print("Percent Fraud in Undersampled Data: ", len(fraudIndices)/len(underSampled_ccfN))
print("Total Cases: ", len(underSampled_ccfN))
print("The largest normalized transaction is $ %.2f" % underSampled_ccfN['nAmount'].max(), ", This is the new car")
print("The smallest normalized transaction is $ %.2f" % underSampled_ccfN['nAmount'].min(), ", This is the new 0")
print("The average normalized transaction is $ %.2f" % underSampled_ccfN['nAmount'].mean(), ", This is the new running shoe")
#It's important to check out the new mean. That is our change in the data due to undersampling
#That is going to cause some form of innacuracy to develop.
#Just a good note, I'm not sure how to deal with it yet.

#Let us train, test, split.
from sklearn.model_selection import train_test_split


#Big data set:
xTtrain, xTest, yTrain, yTest = train_test_split(x,y,test_size=0.333)

#checking lengths to make sure they're right:
print("Big data set:")
print("number of x training: ", len(xTrain))
print("number of y training: ",len(yTrain))

print("number of x testing: ", len(xTest))
print("number of y testing: ",len(yTest))

print("Total x: ", len(xTrain)+len(xTest))
print("Total y: ",len(yTrain)+len(yTest))

print(" ")


#Undersampled data set:
xUTrain, xUTest, yUTrain, yUTest = train_test_split(xU,yU,test_size=0.333)

#checking lengths to make sure they're right:
print("Baby data set:")
print("undersampled number of x training: ", len(xUTrain))
print("undersampled number of y training: ",len(yUTrain))

print("undersampled number of x testing: ", len(xUTest))
print("undersampled number of y testing: ",len(yUTest))

print("undersampled Total x: ", len(xUTrain)+len(xUTest))
print("undersampled Total y: ",len(yUTrain)+len(yUTest))
print(" ")

#Logistic Regression and Classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, precision_recall_curve,auc,roc_auc_score,roc_curve,recall_score,classification_report


#Can't seem to get this next block of code to work:

"""

def getKFoldScores(xData,yData):
    fold = KFold(len(yData),shuffle=False) 


    cParamRange = [0.01,0.1,1,10,100]

    results = pd.DataFrame(index = range(len(cParamRange),2), columns = ['cParam','Mean Recall Score'])
    results['cParam'] = cParamRange

   
    i = 0
    for cParam in cParamRange:
        print('***')
        print('cParam: ', cParam)
        print('***')
        print('')

        recallAccuracies = []
        for iteration, indices in enumerate(fold,start=1):

          
            model = LogisticRegression(C = cParam, penalty = 'l1')


            
            model.fit(xData.iloc[indices[0],:],yData.iloc[indices[0],:].values.ravel())

            
            yUPred = model.predict(xData.iloc[indices[1],:].values)


            recallAccuracy = recall_score(y_train_data.iloc[indices[1],:].values,y_pred_undersample)
            recallAccuracies.append(recall_acc)
            print('Iteration ', iteration,'; Recall Score = ', recallAccuracy)

       
        results.ix[j,'Mean Recall Score'] = np.mean(recallAccuracies)
        i += 1
        print('')
        print('Mean Recall Score ', np.mean(recallAccuracies))
        print('')

    bestC = results.loc[results['Mean Recall Score'].idxmax()]['cParam']
    
  
    print('***')
    print('Best model to choose from cross validation is with C parameter = ', best_c)
    print('***')
    
    return bestC
           

bestC = getKFoldScores(xUTrain,yUTrain)

"""

#Confusion Matrix function:

import itertools
def plotConfusionMatrix (cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Greens):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    tick_marks=np.arange(len(classes))
    plt.xticks(tick_marks,classes,rotation=0)
    plt.yticks(tick_marks,classes)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    else:
        1
        
    thresh=cm.max()/2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j,i, cm[i,j],horizontalalignment='center',color='white' if cm[i,j] > thresh else 'black')
        
    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted label')


#Confusion Matrix on undersamples data:

model = LogisticRegression(C=100,penalty='l1')
model.fit(xUTrain,yUTrain.values.ravel())
yUPred = model.predict(xUTest.values)

confMatrix = confusion_matrix(yUTest,yUPred)
np.set_printoptions(precision=2)

print('Confusion Matrix:', confMatrix[1,1]/(confMatrix[1,0]+confMatrix[1,1]))

classNames=[0,1]
plt.figure()
plotConfusionMatrix(confMatrix,classes=classNames)
plt.show()


#Confusion Matrix on big data set using undersampled data:

model = LogisticRegression(C=10,penalty='l1')
model.fit(xUTrain,yUTrain.values.ravel())
yPred = model.predict(xTest.values)

confMatrix = confusion_matrix(yTest,yPred)
np.set_printoptions(precision=2)

print('Confusion Matrix:', confMatrix[1,1]/(confMatrix[1,0]+confMatrix[1,1]))

classNames=[0,1]
plt.figure()
plotConfusionMatrix(confMatrix,classes=classNames)
plt.show()


#Plotting ROC Curve, I've never done this.

model = LogisticRegression(C=100, penalty='l1')
yUPredScore = model.fit(xUTrain,yUTrain.values.ravel()).decision_function(xUTest.values)

fpr, tpr, thresholds = roc_curve(yUTest.values.ravel(),yUPredScore)
roc_auc = auc(fpr,tpr)
#ROC Curve plot

#plt.title('Recieve Operating Characteristic Curve')
plt.plot(fpr,tpr,'b',label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1], 'g--')
plt.xlim([-0.1,1.0])
plt.ylim([-0.1,1.01])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
