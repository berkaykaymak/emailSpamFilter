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

feature_extractor = TfidfVectorizer(min_df=1, stop_words='english', lowercase='True')



