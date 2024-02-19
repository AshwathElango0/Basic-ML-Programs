import spacy
from spacy import displacy      #useful in rendering

txt = "Apple India has reached an all-time high stock price of 143 dollars this January."

model = spacy.load('en_core_web_sm')        #Pre-trained model

doc = model(txt)                            #Analyzing
dict = {}
for i in doc.ents:
    dict[i]=i.label_                        #Creating dictionary of named entities
print(dict)
