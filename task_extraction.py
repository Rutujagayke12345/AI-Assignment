import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

text = """
Rahul wakes up early every day. He goes to college in the morning and comes back at 3 pm. 
At present, Rahul is outside. He has to buy the snacks for all of us. 
John should submit the report by 5 PM today.
Sarah needs to complete the project by Monday.
"""

def preprocess_text(text):
    sentences = sent_tokenize(text)
    processed_sentences = [pos_tag(word_tokenize(sent)) for sent in sentences]
    return sentences, processed_sentences

def extract_tasks(sentences, tagged_sentences):
    tasks = []
    for i, sentence in enumerate(tagged_sentences):
        for word, tag in sentence:
            if tag.startswith("VB") and ("must" in sentences[i] or "should" in sentences[i] or "has to" in sentences[i] or "needs to" in sentences[i]):
                tasks.append(sentences[i])
                break
    return tasks

sentences, tagged_sentences = preprocess_text(text)
tasks = extract_tasks(sentences, tagged_sentences)

print("Extracted Tasks:")
for task in tasks:
    print("-", task)