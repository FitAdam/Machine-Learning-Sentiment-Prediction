"""Importing libraries"""
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


"""Read the data"""
df = pd.read_csv(r'C:\Users\atuta\Documents\Projects\Stocks Machine Learning\Predictions\Combined_News_DJIA.csv', encoding = 'ISO-8859-1')
df_test = pd.read_csv(r'C:\Users\atuta\Documents\Projects\Stocks Machine Learning\Predictions\3titles.csv', encoding = 'ISO-8859-1')

"""Divide the data into sets"""

#divide in train and test
train = df
test = df_test

"""Prepering data"""

# remove punctuation
# apart from alphabets we will replace every character with blank
data = train.iloc[:, 2:27]
data.replace("[^a-zA-Z]", " ", regex=True, inplace=True)

# Renaming the columns
list1 = [i for i in range(3)]
new_Index = [str(i) for i in list1]
data.columns = new_Index


# converting hadlines to lower case
for index in new_Index:
    data[index] = data[index].str.lower()


' '.join(str(x) for x in data.iloc[1, 0:3])  #Combine all the 3 headlines to a single sentence

#all the sentences in the form of list

headlines = []
for row in range(0, len(data.index)):
    headlines.append(' '.join(str(x) for x in data.iloc[row, 0:3]))

""" Countvectorizer is used to convert all the sentences into vectors """

# implememt BAG OF WORDS
countVector = CountVectorizer(ngram_range=(2,2))
#ngram(2,2) means it will combine the 2 words together and assign the value

trainDataset = countVector.fit_transform(headlines)

"""Preparing testing and training"""

testTransform =[]
for row in range(0, len(test.index)):
    print(f"This is row: {row}")
    testTransform.append(' '.join(str(x) for x in test.iloc[row, 0:3]))
    
test_dataset = countVector.transform(testTransform)

print(f"This is test_dataset: {test_dataset}")

"""Open pickle"""
pickle_open = open("stocks_news_model.pickle", "rb")

"""Load pickle"""
randomForestClassifier = pickle.load(pickle_open)

"""Predict"""
prediction = randomForestClassifier.predict(test_dataset)

""" Print the result """
print(prediction)
prediction = (prediction > 0.59)
print(prediction) # if true the stocks will go up, otherwise down


