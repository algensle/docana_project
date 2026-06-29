import re
import json
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import FreqDist
import sklearn
import spacy
from collections import defaultdict
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import cohen_kappa_score



with open("redditdump.json", "r", encoding="utf-8") as f:
    reddit = json.load(f)

'''
#save raw text in txt file for manual annotation
with open("TempHelper.txt", "w", encoding="utf-8") as f:
    for item in reddit:
        f.write(f"{item} UNGABUNGA \n \n \n") 
'''


texts = []


def extract_bodies_from_comment(comment_data, target_list):
    if "body" in comment_data:
        target_list.append(comment_data["body"])

    replies = comment_data.get("replies")
    if isinstance(replies, dict):
        for reply in replies.get("data", {}).get("children", []):
            reply_data = reply.get("data", {})
            if isinstance(reply_data, dict):
                extract_bodies_from_comment(reply_data, target_list)


for listing in reddit:

    children = listing["data"]["children"]

    for child in children:

        post_data = child["data"]

        extract_bodies_from_comment(post_data, texts)

            
'''
#save text with removed: URLs, markdown links, HTML entities, special characters, and normalize whitespace
with open("TempHelperBody.txt", "w", encoding="utf-8") as f:
    for item in texts:
        f.write(f"{item} UNGABUNGA \n \n \n")  
        
        
with open('TempHelperBody.txt', 'r', encoding='utf-8') as file:
    TempHelperBody= file.read()
'''


    
def clean(dirty):
    dirty = re.sub(r'http\S+', '', dirty)
    dirty = re.sub(r'\[.*?\]\(.*?\)', '', dirty)
    dirty = re.sub(r'&\w+;', '', dirty)
    dirty = re.sub(r'[^\w\s.!?,]', '', dirty)
    dirty = re.sub(r'\s+', ' ', dirty)
    dirty = re.sub(r'UNGABUNGA', '', dirty)
    
    return dirty

cleaned=[]

for instance in texts:
    a=clean(instance)
    cleaned.append(a)
 
'''
#save cleaned text in txt file for manual annotation
with open("TempHelperCleaned.txt", "w", encoding="utf-8") as f:
    for item in cleaned:
        f.write(f"{item} \n \n \n")     
'''   


#create Corpus 
corpus = " ".join(cleaned)
#print(corpus)


#END CLEAN UP



'''
SPACY PART
'''


#initialize spacy 
spacy.prefer_gpu()
nlp = spacy.load("en_core_web_sm")

# Corpus with spaCy 
doc = nlp(corpus)

# seperate sentences (with spaCy)
sentences = list(doc.sents)
firstTenSentences = sentences[:10] #first 10 Sätze
#print(firstTenSentences)

# seperate the sentences in to single words but keep the sentence structure intact
wordsInSentences = [[token.text for token in sent] for sent in sentences]
firstTenTokenized = wordsInSentences[:10] #first 10 Sätze mit Tokenisierung
#print(firstTenTokenized)

# POS Tagging with spaCy (keep sentence structure)
POStaggedWordsInSentences = [[(token.text, token.pos_) for token in sent] for sent in sentences]
firstPOSTaggedWordsInSentences = POStaggedWordsInSentences[:10]#first 10 Sätze mit POS Tags
#print(firstPOSTaggedWordsInSentences)

# Tagging with spaCy (keep sentence structure)
taggedWordsInSentences = [[(token.text, token.tag_) for token in sent] for sent in sentences]
firstTaggedWordsInSentences = taggedWordsInSentences[:10]#first 10 Sätze mit Tags
#print(firstTaggedWordsInSentences)

# Pos extraction (without words)
pos = [token.pos_ for token in doc]
#print(pos[:100])  # print first 100 Pos

# Tags extraction (without words)
tags = [token.tag_ for token in doc]
#print(tags[:100])  # print first 100 Tags

# Frequency Distribution of Tags and Pos-Tags
#fdist_pos = FreqDist(pos)
fdist_tags = FreqDist(tags)
#print(fdist_tags.most_common())
#print(fdist_pos.most_common())



'''
Analysis of the tagged data: amount of sentences, amount of words, mean sentence length
'''

AmountOfWords = len(tags)
AmountOfSentences = len(sentences)
def sentenceLength(sentences):
    raw = [len(sent) for sent in sentences]
    maxLength = max(raw)
    sentenceLengths = [0] * (maxLength + 1)
    for length in raw:
        sentenceLengths[length] += 1
    return sentenceLengths

LengthOfSentences= sentenceLength(sentences) #list of sentence lengths and their frequencies

averageLengthOfSentences = sum(len(sent) for sent in sentences) / len(sentences)

print("The Amount of Words in the Reddit sample is: ", AmountOfWords)
print("The Amount of Sentences in the Reddit sample is: ", AmountOfSentences)
print("The Average Length of Sentences is: ", averageLengthOfSentences )


'''
CSV Export for manual annotation

 #export sample as csv
 import csv
 import random

 #random sample of 40 sentences
 random.seed(42)  # for reproducibility
 sample = random.sample(taggedWordsInSentences, 40)

 #save as csv 
 with open("manual_annotation_sample.csv", "w", newline="", encoding="utf-8") as f:
     writer = csv.writer(f, delimiter=";")
     writer.writerow(["satz_nr", "wort", "auto_tag", "manual_tag"])
     for i, sent in enumerate(sample):
         for word, tag in sent:
             writer.writerow([i, word, tag, ""])  # manual tagging stays empty
         writer.writerow(["", "", "", ""])  # empty row between sentences
'''





'''
Cohen's Kappa
'''

df = pd.read_excel("manual_annotation_sample.xlsx")

#delete spaces
df = df.dropna(subset=["auto_tag", "manual_tag"])
df = df[df["wort"] != ""]

#calculate Kappa score
kappa = cohen_kappa_score(df["auto_tag"], df["manual_tag"])
print(f"Cohen's Kappa: {kappa:.3f}")


'''
Error analysis 
'''

#where do they diver?
disagreements = df[df["auto_tag"] != df["manual_tag"]]
print("Disagreements:")
print(disagreements[["wort", "auto_tag", "manual_tag"]])

#how many deviations?
print(f"Agreements: {(df['auto_tag'] == df['manual_tag']).mean():.1%}")

#only "true" errors without tokenization errors
if "anmerkung" in df.columns:
    df_cleaned = df[df["anmerkung"].isna()]
    kappa_cleaned = cohen_kappa_score(df_cleaned["auto_tag"], df_cleaned["manual_tag"])
    print(f"Cohen's Kappa (cleaned): {kappa_cleaned:.3f}")

#tokenization errors
if "anmerkung" in df.columns:
    tokenization_errors = df[df["anmerkung"].str.contains("tokenerror", na=False)]
    print("Tokenization Errors:")
    print(tokenization_errors[["wort", "auto_tag", "manual_tag", "anmerkung"]])
    print(f"Amount of Tokenization Errors: {len(tokenization_errors)}")
    print(f"Proportion of Tokenization Errors: {len(tokenization_errors) / len(df):.1%}")



#visualisation 

common_tags = fdist_tags.most_common()
tags_all = [t[0] for t in common_tags]
counts_all = [t[1] for t in common_tags]

#relative frequencies for interesting tags
total_tags = sum(counts_all)
for tag in ['MD', 'PRP', 'NNP', 'RB']:
    freq_rel = fdist_tags[tag] / total_tags * 100
    print(f"Relative frequency of {tag}: {freq_rel:.2f}%")

#top 15 tags
tags_top = tags_all[:15]
counts_top = counts_all[:15]


#Frequency Distribution plot: all tags
fig,ax = plt.subplots(figsize = (16,6))
ax.bar(tags_all, counts_all, color ='mediumblue')
ax.set_title('Frequency Distribution of Tags in Redit sample (All Tags)')
ax.set_xlabel('Tag')
ax.set_ylabel('Frequency')
plt.xticks(rotation = 45, ha='right')
plt.tight_layout()
#plt.savefig('freqdist_reddit_all.png', dpi=150)
#plt.show()

#Frequency Distribution plot: top 15 tags
fig,ax = plt.subplots(figsize = (12,6))
ax.bar(tags_top, counts_top, color ='mediumblue')
ax.set_title('Frequency Distribution of Tags in Redit sample (Top 15 Tags)')
ax.set_xlabel('Tag')
ax.set_ylabel('Frequency')
plt.tight_layout()
#plt.savefig('freqdist_reddit_top15.png', dpi=150)
#plt.show()


'''
Transition matrix for Redit sample: Calculates how often each tag follows another tag
'''

#count transitions
transitions = defaultdict(lambda: defaultdict(int))

for sent in taggedWordsInSentences:
    for i in range(len(sent)-1):
        current_tag = sent[i][1]
        next_tag = sent[i+1][1]
        transitions[current_tag][next_tag] += 1
    
#convert to dataframe
tags_total = sorted(set(tags))
transition_df = pd.DataFrame(0, index=tags_total, columns=tags_total)

for current_tag, next_tags in transitions.items():
    for next_tag, count in next_tags.items():
        transition_df.loc[current_tag, next_tag] = count

#normalisation
transition_prob = transition_df.div(transition_df.sum(axis=1), axis=0).fillna(0)


#save all transitions as csv
#transition_prob.to_csv('reddit_transition_matrix.csv', float_format='%.3f', sep=';')

#visualisation for top15 tags
transition_top15 = transition_prob.loc[tags_top, tags_top]

#visualisation as heatmap
fig, ax = plt.subplots(figsize=(12,10))
sns.heatmap(transition_top15,
            cmap = 'Greens',
            ax = ax,
            annot=True,
            fmt='.2f',
            xticklabels=True,
            yticklabels=True)
ax.set_title('Tag Transition Matrix Reddit sample Top 15')
ax.set_xlabel('Next Tag')
ax.set_ylabel('Current Tag')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
#plt.savefig('transition_matrix_reddit.png', dpi=150)
#plt.show()



'''
Package version check

#import sys

print("Package Versions:")
print("nltk:", nltk.__version__)
print("spacy:", spacy.__version__)
print("pandas:", pd.__version__)
print("sklearn:", sklearn.__version__)
print("matplotlib:", plt.matplotlib.__version__)
print("numpy:", np.__version__)
print("seaborn:", sns.__version__)
#print("Python:", sys.version)
'''