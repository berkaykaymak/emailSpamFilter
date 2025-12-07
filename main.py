import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

raw_mail_data = pd.read_csv("mail_data.csv")


mail_data = raw_mail_data.where(pd.notnull(raw_mail_data), '')

#mail_data.head()
#print(mail_data.shape)

#Labeling ham as 1 and  spam as 0

mail_data.loc[mail_data['Category'] == 'spam', 'Category'] = 0
mail_data.loc[mail_data['Category'] == 'ham', 'Category'] = 1


X = mail_data['Message']
Y = mail_data['Category']


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

#print(X.shape)
#print(X_train.shape)
#print(X_test.shape)

#Feature Extraction - transform text data to feature vectors

feature_extractor = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)

#min_def to check repetitive words
#stop_words to ignore some common words
#lowercase to standardization

X_train_features = feature_extractor.fit_transform(X_train)
X_test_features = feature_extractor.transform(X_test)

#convert all data types to int

Y_train= Y_train.astype('int')
Y_test = Y_test.astype('int')

model = LogisticRegression()
model.fit(X_train_features, Y_train)


prediction_on_training_data=model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)

#print(accuracy_on_training_data)
#its around 0.96 so it has accuracy of 96 over 100


prediction_on_test_data=model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)

#print(accuracy_on_test_data)
#its around 0.96 again

