import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from  sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# uploading data ....
path = r'C:\Users\HP\Desktop\Main Folder\ml_ex\data_coll\IMDB Dataset.csv'

dataset = pd.read_csv(path)
print(dataset.head())

from sklearn.feature_extraction.text import TfidfVectorizer

# Replace infinities with NaN first (if needed)
dataset.replace([np.inf, -np.inf], np.nan, inplace=True)

dataset['sentiment']=dataset['sentiment'].map({'positive':1,'negative':0}).astype(int)
X= dataset['review']
Y = dataset.iloc[:,-1]
#print(X)

vectorizer = TfidfVectorizer(stop_words='english')
X_vect = vectorizer.fit_transform(X)

#print(X_vect)
#print(Y)

#Spliting a data..train,test,validation...

X_train,X_test,Y_train,Y_test = train_test_split(X_vect,Y,test_size=0.25,random_state=0)

# Alogrithm loaded....

model=LogisticRegression()

model = model.fit(X_train,Y_train)

# prediction....

Y_pred = model.predict(X_test)
from sklearn.metrics import accuracy_score,confusion_matrix

print((accuracy_score(Y_test,Y_pred))*100)
print(confusion_matrix(Y_test,Y_pred))

#validation

t=[]
for i in range(0,10):

    review = input("enter a yuor reviwew :: ")

    result = model.predict( vectorizer.transform([review]))
    print(result)
    t.append(result)

zero = t.count(0)
one = t.count(1)
if zero<one:
    result_fn=1
elif zero>one:
    result_fn = 0
else:
    result_fn=1

if result_fn ==1:
    print("GOOD MOVIE")
else:
    print("BAD MOVIE")