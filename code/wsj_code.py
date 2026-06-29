#import general modules
import nltk
import json
import os
import re
import spacy
import csv

#for frequency distribution and transition matrix
from nltk import FreqDist
import matplotlib.pyplot as plt
import pandas as pd

from collections import defaultdict
import numpy as np
import seaborn as sns


'''
Pipeline for processing the Penn Treebank data (WSJ)
1. modules nltk, spacy, re for processing and saving
2. load raw text from nltk treebank and save it in a variable
3. clean the text with regex, especially whitespace normalization and save
4. tag it with spacy
5. save as a list of tuples

'''
'''
Preprocessing
'''
#load spacy and treebank
nlp = spacy.load("en_core_web_sm")

#treebank download
#nltk.download('treebank')
from nltk.corpus import treebank_raw

#print(nltk.corpus.treebank.readme())

#load raw text
raw_text = treebank_raw.raw()
#print(type(raw_text))


#cleaning the text with regex
def clean_minimal(text):
    text = re.sub(r'-LRB-', '(', text)
    text = re.sub(r'-RRB-', ')', text)
    text = re.sub(r'-LSB-', '[', text)
    text = re.sub(r'-RSB-', ']', text)

#substitute article start marker with newline
    text = re.sub(r'\.START\s*', '\n', text)
    return text.strip()


#spacy cleaned text, for spacy processing
wsj_spacy = clean_minimal(raw_text).replace('\n', ' ')
    


#tag text with spacy
doc = nlp(wsj_spacy)

#save as a list of tuples
tagged_sentences = []
for sent in doc.sents:
    tagged_sentences.append([(token.text, token.tag_) for token in sent])



## check output
#print(tagged_sentences[0:5])

#amount of sentences
print(f"Number of sentences: {len(tagged_sentences)}")

#amount of tokens
num_tokens = sum(len(sent) for sent in tagged_sentences)
print(f"Number of tokens: {num_tokens}")

#mean sentence length
mean_sentence_length = num_tokens / len(tagged_sentences)
print(f"Mean sentence length: {mean_sentence_length:.2f} tokens per sentence")


#save in csv file
# with open("wsj_tagged.csv", "w", newline='', encoding='utf-8') as f:
#     writer = csv.writer(f)
#     writer.writerow(["sentence_nr", "token", "pos_tag"])
#     for i, sent in enumerate(tagged_sentences):
#         for word, tag in sent:
#             writer.writerow([i, word, tag])



'''
Frequency distribution of POS tags in the WSJ data
'''

# extraction of POS tags from the tagged sentences
tags_wsj = [tag for sent in tagged_sentences for (word, tag) in sent    ]

#create frequency distribution of POS tags
fdist_wsj = FreqDist(tags_wsj)
common_tags = fdist_wsj.most_common()
#print(common_tags)


#all tags
tags_all = [t[0] for t in common_tags]
counts_all = [t[1] for t in common_tags]

#relative frequencies for interesting tags
total_tags = sum(counts_all)
for tag in ['MD', 'PRP', 'NNP', 'RB']:
    freq_rel = fdist_wsj[tag] / total_tags * 100
    print(f"Relative frequency of {tag}: {freq_rel:.2f}%")

#top 15 tags
tags_top = tags_all[:15]
counts_top = counts_all[:15]

#Frequency Distribution plot: all tags
fig,ax = plt.subplots(figsize = (16,6))
ax.bar(tags_all, counts_all, color ='steelblue')
ax.set_title('Frequency Distribution of POS Tags in WSJ (All Tags)')
ax.set_xlabel('POS Tag')
ax.set_ylabel('Frequency')
plt.xticks(rotation = 45, ha='right')
plt.tight_layout()
#plt.savefig('freqdist_wsj_all.png', dpi=150)
#plt.show()

#Frequency Distribution plot: top 15 tags
fig,ax = plt.subplots(figsize = (12,6))
ax.bar(tags_top, counts_top, color ='steelblue')
ax.set_title('Frequency Distribution of POS Tags in WSJ (Top 15 Tags)')
ax.set_xlabel('POS Tag')
ax.set_ylabel('Frequency')
plt.tight_layout()
#plt.savefig('freqdist_wsj_top15.png', dpi=150)
#plt.show()

'''
Transition matrix for WSJ data: Calculates how often each tag follows another tag
'''


#count transitions
transitions = defaultdict(lambda: defaultdict(int))

for sent in tagged_sentences:
    for i in range(len(sent)-1):
        current_tag = sent[i][1]
        next_tag = sent[i+1][1]
        transitions[current_tag][next_tag] += 1
    
#convert to dataframe
tags_total = sorted(set(tags_wsj))
transition_df = pd.DataFrame(0, index=tags_total, columns=tags_total)

for current_tag, next_tags in transitions.items():
    for next_tag, count in next_tags.items():
        transition_df.loc[current_tag, next_tag] = count

#normalisation
transition_prob = transition_df.div(transition_df.sum(axis=1), axis=0).fillna(0)


#save all transitions as csv
#transition_prob.to_csv('wsj_transition_matrix.csv', float_format='%.3f', sep=';')

#visualisation for top15 tags
transition_top15 = transition_prob.loc[tags_top, tags_top]

#visualisation as heatmap
fig, ax = plt.subplots(figsize=(12,10))
sns.heatmap(transition_top15,
            cmap = 'Blues',
            ax = ax,
            annot=True,
            fmt='.2f',
            xticklabels=True,
            yticklabels=True)
ax.set_title('POS Tag Transition Matrix WSJ Corpus Top 15')
ax.set_xlabel('Next POS Tag')
ax.set_ylabel('Current POS Tag')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
#plt.savefig('transition_matrix_wsj.png', dpi=150)
#plt.show()

