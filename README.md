# Register Variation in Peer Discussions vs. News Articles

_Group members: Anna-Lena Gensle, Sebastian Berl_

## Introduction

Language strongly varies across contexts, with spoken and written language differing particularly in register and style. Online-communication and journalism are two very different written contexts, although online-communication tends to adapt features of spoken language. Register differ in the situation specific speech variety: while in formal written language a nominal style with high lexical density is common, in informal written language a verbal style with more pronouns and modal verbs is used (Biber & Conrad, 2009).  
NLP models are often trained on formal written texts but used on informal ones which is a well known issue. Part-of-speech (POS) distributions are a measurable feature of these differences.  
Related work has investigated POS tagging challenges in informal social media text. Li & Liu (2015) have jointly modelled text normalization and POS tagging for Twitter data and found that especially non-standard tokens pose a serious problem for POS tagging accuracy, highlighting the importance to account for register-specific challenges when applying POS taggers to different text types.

Keeping this in mind, we are interested in investigating the different usage of POS tags in informal and formal written text. We try to answer the question: How does the distribution of word classes change in peer discussions in comparison to news articles? 
To address this question our research questions are the following:

### Research Questions: 

1. How does the POS tag frequency distribution differ between Reddit and the Wall Street Journal (WSJ)?
2. How do the transition probabilities of the POS tags differ between the two corpora?
3. Do the observed differences reflect the expected register features?



## Dataset
Since we wanted to compare formal and informal written text for our analysis, we decided to use the following two datasets:

### Reddit Corpus
As a dataset for our informal written text we used Reddit as our source. We only used Reddit threates and discussions in which political subjects were discussed. [Reddit - 2016 Presidential Debate Discussion](https://www.reddit.com/r/PoliticalDiscussion/comments/54nuut/debate_megathread_live_discussion_thread_for_the/)

The amount of tokens in the Reddit sample is  18.655, the amount of sentences in the Reddit sample is  1.177.

### WSJ Corpus
The Wall Street Journal (WSJ) corpus is part of the Penn Treebank (Marcus et al., 1993) and consists of journalistic text, covering business, finance and politics. For the project we used the unlabeled version which is available via NLTK:
```python
import nltk
from nltk.corpus import treebank_raw

nltk.download('treebank')
raw_text = treebank_raw.raw()
```
This sample contains a fragment of around 5% of the original Treebank, covering 1.650 sentences from the 1989 Wall Street Journal articles, distributed under fair use for non-commercial, illustrative purposes (Penn Treebank README, LDC 1995).
After cleaning and tagging the raw text with spaCy, the processed corpus consisted of 3.849 sentences and 98.492 tokens.
## Methods

### Setup 

Our project uses the Python version 3.9.25 . It is mandatory to use this version specifically since some packages we used will not run if a newer version is selected. As tokenizer and tagger we used spaCy 3.7.6 with the en_core_web_sm model. In our code we used several packages to analyse our two data sets, some of them were used to ensure a smooth import of the data like json and others like spaCy for analysing purposes. We provided a full overview of all packages and versions in the  `requirements.txt` file. 

### Experiments

### Reddit

#### Preprocessing
We first chose suitable discussions on reddit. The discussions should be around a political topic.

After choosing suitable subreddit discussions, we started out by cleaning the reddit sample. Therefore we removed all emojis, URL links, markdown-links, HTML entetiesa and specific signs like ": ; ( ) [ ] { } / \ @ # $ % ^ & * - + = < >". After completing this, we formated our data in many different ways and tagged them using spaCy to be able to create our distribution matrixes and frequency diagrams. 

#### Manual Annotation & Cohen's Kappa
Since spaCy was primarily trained on formal written texts and the reddit texts are informal, we weren't sure how well the tagger would perform on this data set and if the results were therfore reliable. Hence, we did a manual annotation and evaluated a random sample of 40 sentences using random.seed(42) for reproducibility. The manual tagging was performed in an excel spreadsheet with the Penn Treebank tagset and anotator agreement was evaluated with the Cohen's kappa measure. The results of the manual tagging were the following:

The initial Cohen's Kappa was 0.9888, the gold standard, so the maximum possible value, is 1.0 . So overall, we saw a good performance except for some internet specific interjections which were labeled as proper nouns like "lmao". But what came to our attention were several tokenization errors especially for abbreviated verb forms like "I'll". Therefore we decided to differentiate between tokenization errors and tagging errors which lead to a cleaned Cohen's Kappa of 0.991 and a proportion of tokenization errors of 0.8%.

Tokenization Errors:

| Word  | Auto Tag | Manual Tag | Note       |
|-------|----------|------------|------------|
| its   | PRP$     | PRP$       | tokenerror |
| Its   | PRP$     | PRP$       | tokenerror |
| under | IN       | JJ         | tokenerror |
| Ill   | NNP      | PRP        | tokenerror |
| its   | PRP$     | PRP$       | tokenerror |

Amount of Tokenization Errors: 5
Proportion of Tokenization Errors: 0.8%

Overall, spaCy performed well on the reddit data set with a near gold standard Cohen's kappa of 0.991, but it had problems with abbreviated verb forms and typical online communication interjections. Nevertheless, this evaluation accounts for a better reliability of our results.

### WSJ Corpus

#### Preprocessing
We extracted the raw text from the Penn Treebank as described above and saved it in a variable as string. For cleaning it we used a function with regular expressions substituting the corpus inherent artefacts with actual brackets,substituting the raw text inherent start label with a newline to keep it readable and then replaced the newline for spaCy processing.
```python
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
```
We tagged the text with spaCy which tokenizes and POS tags at the same time. We stored the results as a list of tuples (token,tag).
```python
#tag text with spacy
doc = nlp(wsj_spacy)

#save as a list of tuples
tagged_sentences = []
for sent in doc.sents:
    tagged_sentences.append([(token.text, token.tag_) for token in sent])
```
#### POS tagging
Both corpora were tagged using the same pipeline to ensure comparability, as illustrated in the previous section. We used spaCy's token_tag_ attribute which results in Penn Treebank tags, since spaCy differs in POS attribute and tag attribute.

#### Frequency distribution
To analyse the distribution of POS tags in both corpora, we extracted all tags from the tagged sentences and computed a frequency distribution using NLTK's FreqDist. Since the corpora differ in size we additionally calculated relative frequencies for selected tags to allow a fair comparison.

#### Transition Matrix
To capture patterns between POS tags, we computed a transition matrix for both corpora. For each pair of consecutive tags in the tagged sentences, we counted how often one tag follows another and normalized the counts row-wise to obtain probabilities. For example a value of 0.4 in row NN and column IN therefore means that in 40% of the cases a preposition follows a noun.

## Results and Discussion
### Frequency Distribution

![WSJ Frequency Distribution](figures/freqdist_wsj_top15.png)

![Reddit Frequency Distribution](figures/freqdist_reddit_top15.png)

Both corpora contain on the most frequent positions singular nouns and prepositions but also determiners. While the WSJ corpus contains more proper nouns and plural nouns the reddit sample contains more adverbs and personal pronouns on the more frequent positions. What stands out is that the WSJ corpus contains a lot more cardinal numbers than the reddit sample while the reddit sample contains interjections and the WSJ corpus barely does. Modal verbs are more frequent in the reddit sample (1.53%) than in the WSJ corpus (0.94%).

| POS Tag | WSJ | Reddit |
|---------|-----|--------|
| MD (Modal verbs) | 0.94% | 1.53% |
| PRP (Personal pronouns) | 1.74% | 7.08% |
| NNP (Proper nouns) | 9.87% | 4.62% |
| RB (Adverbs) | 3.07% | 7.72% |

So more singular and proper nouns in the WSJ corpus align with a nominal style with high lexical density. More pronouns and modals in the reddit sample align with a verbal style and a more personal address which both fits expected register features. More adverbs in the reddit samples reflect intensified speech which is typical for online-communication. The WSJ contains much more proper nouns and cardinals since it's typical in journalistic convention to describe developments, companies and names more specific than it would be in reddit discussions. 
However, since the corpora differ considerably in size (98.492 vs. 18.655 tokens), the comparability of the results may be effected by this difference, despite of using relative frequencies.
To conclude this section, the distributions align with the register features described by Biber & Conrad (2009), the WSJ shows a more nominal style with high lexical density and the Reddit sample a more verbal, more personal style.

### Transition Probabilities

![WSJ Transition Matrix](figures/transition_matrix_wsj.png)

![Reddit Transition Matrix](figures/transition_matrix_reddit.png)

The following table highlights the most notable differences in transition probabilities between the two corpora:

| Transition | WSJ | Reddit |
|------------|-----|--------|
| DT → NN | 0.47 | 0.42 |
| NNP → NNP | 0.38 | 0.11 |
| PRP → VBP | 0.18 | 0.27 |

The highest transition probabilities in both corpora are from determiner to noun and adjective to noun. What is interesting here is that in the WSJ corpus nouns are more often directly preceded by determiners (0.47) while in the Reddit sample this direct transition is smaller (0.42) and adjectives are more often in between the noun and determiner than in the WSJ, which fits a more descriptive style in the Reddit sample. In the WSJ corpus there are higher transition probabilities from past tense verbs to prepositions and from proper noun to proper noun. This high transition probability from proper noun to proper noun can be explained since there are proper names with several parts for companies or institutions, the high transition from past tense forms to prepositions suggest a more complex syntactic structure. Both characteristics are typical for journalistic writing. In the reddit sample, there are higher transition probabilities from prepositions to determiners and personal pronouns to verbforms in non 3rd person singular. This suggests a more direct, more personal writing style. In the reddit sample the transition probability for verb non 3rd person singular to adverb is also higher while these are not in the top15 tags from WSJ.

Overall, the observed differences both in frequency distributions and transition probabilities reflect the register features predicted by Biber & Conrad (2009). The WSJ shows a nominal, institunionally oriented style, which is typical for formal journalism, while the reddit sample shows a more verbal, interpersonal style, which is characteristic of informal online communication.

## Conclusion

Our goal of this project was to compare the register of informal written text and formal written text, analysing the POS distribution and transition probabilities between the Reddit Corpus and the Wall Street Journal from the Penn Treebank Project. Both corpora were preprocessed in a similar pipeline and annotated with the same tagger, spaCy, to obtain comparable results. An additional manual evaluation on the Reddit dataset was conducted to account for possible erros of the automatic tagger. Frequency distributions and transition matrices were calculated to measure the different usage of POS tags in both datasets.

The main findings were that for the frequency distribution, a nominal style for the WSJ corpus could be observed in the POS tag distribution and a verbal style in the Reddit corpus, which aligns with the predictions by Biber & Conrad (2009). The transition matrices showed high NNP to NNP transitions and more complex verb structures for the WSJ corpus while the Reddit corpus showed more PRP to VBP transitions which aligns with a more personal writing style. So both corpora analyses show consistent register specific differences.

With respect to our research questions, for RQ 1 we can observe clear frequency differences in the POS tag distribution and therefore confirm our first research question. We can confirm our RQ2 as well since the transition matrixes showed different syntactic transition patterns. These findings lead to the third research question, if the observed results reflect the expected register differences. From a result based point of view we can confirm this, as the observed differences align with the register features predicted by Biber & Conrad (2009). Nevertheless, there are several limitations which need to be mentioned:
The first and probably most important limitation is the corpus size of our datasets. 98.492 WSJ tokens vs. 18.655 Reddit tokens is a notable limitation of our work, despite using relative numbers for the analysis. This should be kept in mind when interpreting the results.
A second limitation would be that we manually annotated only the Reddit corpus because we expected an reduced performance of the tagger with this dataset, but didn't check the performance on the WSJ corpus, because we expected it to be good. Another limitation would be that the corpora differ in their publishing year and we therefore need to consider change in language over the years since the late 1980s.
So for a future comparison larger and more even corpora would enhance the comparability. On top, manual annotations for the WSJ could improve the validity of the results and more registers could be compared.

## Contributions

| Team Member       | Contributions                                               |
|-------------------|-------------------------------------------------------------|
| Anna-Lena Gensle  | WSJ corpus: Data collection, preprocessing, corpus analysis, evaluation, manual annotation Reddit and Cohen's kappa |                                                       
| Sebastian Berl    | Reddit Corpus: Data collection, preprocessing, corpus analysis, evaluation |

## References

Biber, D., & Conrad, S. (2009). *Register, genre, and style*. Cambridge University Press.

Li, C., & Liu, Y. (2015). Joint POS tagging and text normalization for informal text. In *Proceedings of the Twenty-Fourth International Joint Conference on Artificial Intelligence (IJCAI 2015)* (pp. 1263–1269). https://www.ijcai.org/Proceedings/15/Papers/182.pdf

Marcus, M. P., Santorini, B., & Marcinkiewicz, M. A. (1993). Building a large annotated corpus of English: The Penn Treebank. *Computational Linguistics*, 19(2), 313–330. https://aclanthology.org/J93-2004/
