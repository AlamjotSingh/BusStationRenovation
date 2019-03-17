
# coding: utf-8

# In[15]:


import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
import graphviz 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


def importdata():
    data=pd.read_csv("E:\\Datasets\\Auction_new.csv", header=0)
    return data
def splitdataset(data):
    X=data.values[:,1:8]
    Y=data.values[:,8]
    X_train, X_test, Y_train, Y_test=train_test_split(X,Y, train_size=40, test_size=16, random_state=1)
    return X, Y, X_train, X_test, Y_train, Y_test
def train_using_infogain(X_train, X_test, Y_train):
    clf_gain = DecisionTreeClassifier(
            criterion = "entropy", random_state = 100)
    clf_gain.fit(X_train, Y_train)
    return clf_gain
def classify(X_test, clf_gain):
 
    # Predicton on test with giniIndex
    Y_pred = clf_gain.predict(X_test)
    #print("Predicted values:")
    #print(Y_pred)
    return Y_pred
    Y_pred_gain_25 = classify(X_test, clf_gain_25)
def visualize(data, clf_gain):    
  #  graph = graphviz.Source( tree.export_graphviz(clf_gain, out_file=None,
                     #       class_names = clf_gain.classes_,feature_names = data.columns[1:6],filled=True, rounded=True))
    graph = graphviz.Source( tree.export_graphviz(clf_gain, class_names = clf_gain.classes_, 
                                                  out_file=None,filled=True, feature_names=data.columns[1:8]))

    graph.format = 'png'
    graph.render('dtree_render',view=True)
def cal_accuracy(y_test, y_pred, a):
    print("Analysis of tree having min_samples_leaf ",a,".......")
    print("Confusion Matrix : ",
    confusion_matrix(y_test, y_pred))
     
    print ("Accuracy : ",
    accuracy_score(y_test,y_pred)*100)
     
    print("Report : ",
    classification_report(y_test, y_pred))
    
#Driver code
def main():
    data=importdata()
    X, Y, X_train, X_test, Y_train, Y_test=splitdataset(data)
    clf_gain_5 = train_using_infogain(X_train, X_test, Y_train)
    print("Results Using Entropy:")
    # Prediction using entropy
    Y_pred_gain_5 = classify(X_test, clf_gain_5)
    visualize(data, clf_gain_5)
    
    cal_accuracy(Y_test, Y_pred_gain_5,5)
   # tree.export_graphviz(clf_gain_5,out_file='tree.dot')  
#main funcion calling
if __name__=="__main__":
 main()


