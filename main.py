import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from email_fetcher import get_latest_email_content


MY_EMAIL = "........@gmail.com"
MY_APP_PASSWORD = ""


raw_mail_data = pd.read_csv("mail_data.csv")
mail_data = raw_mail_data.where(pd.notnull(raw_mail_data), '')


mail_data['Category'] = mail_data['Category'].map({'spam': 0, 'ham': 1})

X = mail_data['Message']
Y = mail_data['Category']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)


feature_extractor = TfidfVectorizer(min_df=1, lowercase=True)

X_train_features = feature_extractor.fit_transform(X_train)
X_test_features = feature_extractor.transform(X_test)

Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')


print("Training...")
model = LogisticRegression(class_weight='balanced')
model.fit(X_train_features, Y_train)

accuracy = accuracy_score(Y_test, model.predict(X_test_features))
print(f"Test Accuracy: {accuracy:.2f}")
print("-" * 40)


print("Connecting to mailbox...")
input_text = get_latest_email_content(MY_EMAIL, MY_APP_PASSWORD)

if input_text:
    print(f"\n--- INCOMING MAIL ---\n{input_text}\n------------------")


    input_data_features = feature_extractor.transform([input_text])


    probabilities = model.predict_proba(input_data_features)
    spam_probability = probabilities[0][0]  # Spam (0) olma ihtimali
    ham_probability = probabilities[0][1]  # Ham (1) olma ihtimali

    print(f"\n Detailed Analysis:")
    print(f"Spam Chance: %{spam_probability * 100:.2f}")
    print(f"Ham Chance:  %{ham_probability * 100:.2f}")


    # If its over 50%, its a spam.
    threshold = 0.50

    if spam_probability > threshold:
        print("\nIt is a spam mail! Be careful.")
    else:
        print("\nThis mail is not a spam.")

else:
    print("Could not read")