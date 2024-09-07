import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

df = pd.read_csv('spam.csv',encoding = 'latin-1')

del df['Unnamed: 2']
del df['Unnamed: 3']
del df['Unnamed: 4']

df.columns = ['category','message']
l = []
for i,c,m in df.itertuples():
    l.append(len(m))
df.insert(2,'length',l)

df["category"].value_counts().plot(kind = 'pie', explode = [0, 0.1], figsize = (6, 6), autopct = '%1.1f%%', shadow = True)
plt.ylabel("Spam vs Ham")
plt.legend(["Ham", "Spam"])
plt.show()

X,y = df['message'],df['category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

plt.figure(1, figsize = (10, 10))
plt.subplot(121)
y_train.value_counts().plot(kind = 'pie', explode = [0, 0.1], autopct = '%1.1f%%', shadow = True)
plt.ylabel("Spam vs Ham")
plt.xlabel('train data')
plt.legend(["Ham", "Spam"])
plt.subplot(122)
y_test.value_counts().plot(kind = 'pie', explode = [0, 0.1], autopct = '%1.1f%%', shadow = True)
plt.ylabel("Spam vs Ham")
plt.xlabel('test data')
plt.legend(["Ham", "Spam"])
plt.subplots_adjust(wspace=0.5)
plt.show()

text_clf = Pipeline([('tfidf',TfidfVectorizer()),('svm',LinearSVC())])

text_clf.fit(X_train,y_train)

predictions = text_clf.predict(X_test)

print('\n\n', 'Confusion Matrix:')
print(confusion_matrix(y_test, predictions))

print('\n\n*************************************************\n\n')

print(classification_report(y_test, predictions))

print('\n\n*************************************************\n\n')

acc = accuracy_score(y_test, predictions)
acc = round((acc*100),2)
print('Accuracy: ',acc,'%', '\n\n')

msg = [input('enter a text: ')]

result = text_clf.predict(msg)[0]
print('The text is a ', result)
