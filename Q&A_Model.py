import json         #necessary modules
import transformers

with open("", 'rb') as f:       #loading data
    squad = json.load(f)
from transformers import BertTokenizer, BertForQuestionAnswering        #Bert models and tokenizers for said models

model_name = 'deepset/bert-base-cased-squad2'

tokenizer = BertTokenizer.from_pretrained(model_name)       #selecting model and tokenizer
model = BertForQuestionAnswering.from_pretrained(model_name)

qa = transformers.pipeline('question-answering', model=model, tokenizer=tokenizer)  #using pre-built pipeline
          
answers = []

for pair in squad[:7]:     #choosing a number of questions to have answered
    answer = qa({'question' : pair['question'], 'context' : pair['context']})
    answers.append(answer['answer'])              #creating a list of answers to questions in order

for i in answers:
    print(i)



