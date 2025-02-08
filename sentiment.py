from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

reviews = [
    ("I love this product, it's amazing!", "positive"),
    ("This is the worst purchase I've made.", "negative"),
    ("Highly recommend this item, great quality.", "positive"),
    ("Terrible experience, not worth the money.", "negative"),
    ("Fantastic service and fast delivery!", "positive"),
    ("The product broke within a week, disappointed.", "negative"),
]

texts, labels = zip(*reviews)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)
y = [1 if label == "positive" else 0 for label in labels]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
