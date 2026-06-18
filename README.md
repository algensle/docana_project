
## Project Report Template

> This repository serves as a template for your project reports as part of the Document Analysis lecture. To set up your project report as a webpage using GitHub Pages, simply follow the steps outlined in the next chapter.
>
>**Some Organizational Details:** Get creative with your project ideas! Just make sure they relate to Natural Language Processing and incorporate this specified dataset: [Link to data](https://huggingface.co/datasets/webis/tldr-17), [Link to paper](https://aclanthology.org/W17-4508.pdf). Submissions should be made in teams of 2-3 students. Each team is expected to create a blog-style project website, using GitHub Pages, to present their findings. Additionally, teams will deliver a lightning talk during the final lecture to discuss their project. Add all your code, such as Python scripts and Jupyter notebooks, to the `code` folder. Use markdown files for your project report. [Here](https://docs.gitlab.com/ee/user/markdown.html) you can read about how to format Markdown documents. 
>
>Have fun working on your project! 🥳

## Setup The Report Template

Follow this steps to set up your project report:

1. **Fork the Repository:** Begin by creating a copy of this repository for your own use. Click the `Fork` button at the top right corner of this page to do this.

2. **Configure GitHub Pages:** Navigate to `Settings` -> `Pages` in your newly forked repository. Under the `Branch` section, change from `None` to `master` and then click `Save`.

3. **Customize Configuration:** Modify the `_config.yml` file within your repository to personalize your site. Update the `title:` to reflect the title of your project and adjust the `description:` to provide a brief summary.

4. **Start Writing:** Start writing your report by modifying the `README.md`. You can also add new Markdown files for additional pages by modifying the `_config.yml` file. Use the standard [GitHub Markdown syntax](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax) for formatting. 

5. **Access Your Site:** Return to `Settings` -> `Pages` in your repository to find the URL to your live site. It typically takes a few minutes for GitHub Pages to build and publish your site after updates. The URL to access your live site follows this schema: `https://<<username>>.github.io/<<repository_name>>/`

***

# Register Variation in Peer Discussions vs. News Articles

_Group members: Anna-Lena Gensle, Sebastian Berl_

## Introduction

Language strongly varies across contexts, with spoken and written language differing particularly in register and style. Online-communication and journalism are two very different written contexts, although online-communication tends to adapt features of spoken language. Register differ in the situation specific speech variety: while in formal written language a nominal style with high lexical density is common, in informal written language a verbal style with more pronouns and modal verbs is used (Biber & Conrad 2009).  
NLP models are often trained on formal written texts but used on informal ones which is a well known issue. Part-of-speech (POS) distributions are a measurable feature of these differences.  
Related work has investigated POS tagging challenges in informal social media text. Li & Liu (2015) have jointly modelled text normalization and POS tagging for Twitter data and found that especially non-standard tokens pose a serious problem for POS tagging accuracy, highlighting the importance to account for register-specific challenges when applying POS taggers to different text types.

Keeping this in mind, we are interested in investigating the different usage of POS tags in informal and formal written text. We try to answer the question: How does the distribution of word classes change in peer discussions in comparison to news articles? 
To address this question our research questions are the following:

### Research Questions: 

1. How does the POS tag frequency distribution differ between Reddit (r/politics, r/news) and the Wall Street Journal (WSJ)?
2. How do the transition probabilities of the POS tags differ between the two corpora?
3. Do the observed differences reflect the expected register features?



## Dataset
Since we wanted to compare formal and informal written text for our analysis, we decided to use the following two datasets:

### Reddit Corpus
SEBASTIAN
Provide a short description of the dataset used in your project. Focus on highlighting the aspects that are particularly relevant to your work.

### WSJ Corpus
The Wall Street Journal (WSJ) corpus is part of the Penn Treebank (Marcus et al. 1993) and consists of journalistic text, covering business, finance and politics. For the project we used the unlabeled version which is available via NLTK:
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

Our project was coded in the conda-environment "docana" as described in the beginning of the course with Python v.3.9. We used VS Code as IDE and Gitlab for collaborative work. As tokenizer and tagger we used spaCy 3.7.6 with the en_core_web_sm model.

```bash
conda create --name docana python=3.9
conda activate docana
```

```python
nltp = spacy.load("en_core_web_sm")
```

SEBASTIAN HILFE WAS MUSS DA REIN?
Outline the tools, software, and hardware environment, along with configurations used for conducting your experiments. Be sure to document the Python version and other dependencies clearly. Provide step-by-step instructions on how to recreate your environment, ensuring anyone can replicate your setup with ease:



Include a `requirements.txt` file in your project repository. This file should list all the Python libraries and their versions needed to run the project. Provide instructions on how to install these dependencies using pip, for example:

```bash
pip install -r requirements.txt
```

### Experiments

Report how you conducted the experiments. We suggest including detailed explanations of the preprocessing steps and model training in your project. For the preprocessing, describe  data cleaning, normalization, or transformation steps you applied to prepare the dataset, along with the reasons for choosing these methods. In the section on model training, explain the methodologies and algorithms you used, detail the parameter settings and training protocols, and describe any measures taken to ensure the validity of the models.

### Reddit
SEBASTIAN
#### Preprocessing

#### Manual Annotation & Cohen's Kappa
ANNA

#### Frequency Distribution

#### Transition Matrix

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
#### Frequency distribution

#### Transition Matrix
## Results and Discussion

Present the findings from your experiments, supported by visual or statistical evidence. Discuss how these results address your main research question.

## Conclusion

Summarize the major outcomes of your project, reflect on the research findings, and clearly state the conclusions you've drawn from the study.

## Contributions

| Team Member  | Contributions                                             |
|--------------|-----------------------------------------------------------|
| Alice Smith  | Data collection, preprocessing, model training, evaluation|                                                       |
| Bob Johnson  | ...                                                       |
| ...          | ...                                                       |

## References

Include a list of academic and professional sources you cited in your report, using an appropriate citation format to ensure clarity and proper attribution.

