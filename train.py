# Decision Tree Classification

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#define dataset
ds_file = 'files_for_training_model/combined_dataset.csv'

# Importing the dataset
dataset = pd.read_csv(ds_file, sep=",", on_bad_lines='skip', index_col=False)

# Filter out the rows on dataset where value on column TCP_Length is 0
dataset.drop(dataset.loc[dataset['TCP_Length']== 0].index, inplace=True)

# Define website classes column and selected features column
selectedFeatures = dataset.iloc [:, 4:6].values
websiteClass = dataset.iloc [:,7].values

# Separate the dataset into the Training set and Test set
# using 8:2 or 80% training and 20% testing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(selectedFeatures, websiteClass, test_size = 0.20, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', splitter = 'best', random_state = 0)
classifier.fit(x_train, y_train)

# Pickle the trained model
import pickle
pickle.dump(classifier, open('models/classifier.pkl', 'wb'))

# get the model accuracy score using the training data
print(classifier.score(x_train, y_train))

# Predicting the Test set results
y_predResult = classifier.predict(x_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predResult)

# get classifier accuracy
#from sklearn.metrics import accuracy_score
#print("Accuracy: ", accuracy_score(y_test, y_predResult))

# Predicting the single row of test set result
#y_predResultSingle = classifier.predict([[100, 2]])
#print(y_predResultSingle)


# Verifiy that pickled model is work properly 
# with dummy values for frame length (100) and tcp length (100)
classifier = pickle.load(open('models/classifier.pkl', 'rb'))
print(classifier.predict([[100, 100]]))
for i in classifier.predict([[100, 100]]):
    print(i)


