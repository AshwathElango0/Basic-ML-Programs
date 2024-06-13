import requests
import os

squad_dir = ''      #Enter directory file has to be stored in
url = 'https://rajpurkar.github.io/SQuAD-explorer/dataset/'
file = 'dev-v2.0.json'

res = requests.get(url+file)
    # write to file in chunks
with open(os.path.join(squad_dir, file), 'wb') as f:
    for chunk in res.iter_content(chunk_size=40):
        f.write(chunk)

import json         #useful to load data in json format

with open(os.path.join(squad_dir, 'dev-v2.0.json'), 'rb') as f:
    squad = json.load(f)                    #loading data

new_squad = []


for group in squad['data']:                     #navigating the format of stored data
    for paragraph in group['paragraphs']:
        context = paragraph['context']
        for qa_pair in paragraph['qas']:
            question = qa_pair['question']
            if 'answers' in qa_pair.keys() and len(qa_pair['answers']) > 0:
                answer = qa_pair['answers'][0]['text']
            elif 'plausible_answers' in qa_pair.keys() and len(qa_pair['plausible_answers']) > 0:
                answer = qa_pair['plausible_answers'][0]['text']
            else:
                answer = None
            new_squad.append({                  #finding questions and answers
                'question': question,
                'answer': answer,
                'context': context
            })

