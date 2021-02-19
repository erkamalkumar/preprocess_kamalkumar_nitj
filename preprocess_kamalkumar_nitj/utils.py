import re
import os
import sys
import pandas as pd
import numpy as np 
import spacy
from spacy.lang.en.stop_words import STOP_WORDS as stopwords
from bs4 import BeautifulSoup
import unicodedata
from textblob import TextBlob
def _get_wordcounts(x):
	length = len(str(x).split())
	return length

def _get_charcounts(x):
	s = x.split()
	x = ''.join(s)
	return len(x)

def _get_avg_wordlength(x):
	count = _get_charcounts(x)/_get_wordcounts(x)
	return count

def _get_stopwords_counts(x):
	l = len([t for t in x.split() if t in stopwords])
	return l

def _get_hashtag_counts(x):
	l = len([t for t in x.split() if t.startswith('#')])
	return l

def _get_mentions_counts(x):
	l = len([t for t in x.split() if t.startswith('@')])
	return l

def _get_digit_counts(x):
	return len([t for t in x.split() if t.isdigit()])

def _get_uppercase_counts(x):
	return len([t for t in x.split() if t.isupper()])

def _get_cont_exp(x):
	contractions = {
	"'aight" : "alright",
	"amn't" : "am not",
	"aren't" : "are not",
	"can't" : "can not",
	"'cause" : "because",
	"could've":	"could have",
	"couldn't":	"could not",
	"couldn't've":	"could not have",
	"daren't":	"dare not / dared not",
	"daresn't":	"dare not",
	"dasn't":	"dare not",
	"didn't":	"did not",
	"doesn't":	"does not",
	"don't"	:"do not", 
	"dunno":	"don't know",
	"d'ye":	"do you",
	"e'er":	"ever",
	"'em":	"them",
	"everybody's":	"everybody is",
	"everyone's":	"everyone is",
	"finna":	"fixing to",
	"g'day":	"good day",
	"gimme":	"give me",
	"giv'n":	"given",
	"gonna":	"going to",
	"gon't":	"go not",
	"gotta":	"got to",
	"hadn't":	"had not",
	"had've":	"had have",
	"hasn't":	"has not",
	"haven't":	"have not",
	"he'd":	"he had",
	"he'll":	"he shall / he will",
	"he's":	"he has / he is",
	"here's":	"here is",
	"he've":	"he have",
	"how'd":	"how did / how would",
	"howdy":	"how do you do",
	"how'll":	"how will",
	"how're":	"how are",
	"I'd":	"I had",
	"I'd've":	"I would have",
	"I'll": "I will",
	"I'm":	"I am",
	"I'm'a":	"I am about to",
	"I'm'o":	"I am going to",
	"innit":	"is it not",
	"I've":	"I have",
	"isn't":	"is not",
	"it'd":	"it would",
	"it'll": "it will",
	"it's":	"it has",
	"iunno":	"I don't know",
	"kinda":	"kind of",
	"let's":	"let us",
	"ma'am":	"madam",
	"mayn't":	"may not",
	"may've":	"may have",
	"methinks":	"me thinks",
	"mightn't":	"might not",
	"might've":	"might have",
	"mustn't":	"must not",
	"mustn't've":	"must not have",
	"must've":	"must have",
	"needn't":	"need not",
	"nal":	"and all",
	"ne'er":	"never",
	"o'clock":	"of the clock",
	"o'er":	"over",
	"ol'":	"old",
	"oughtn't":	"ought not",
	"shalln't":	"shall not",
	"shan't":	"shall not",
	"she'd":	"she had",
	"she'll":	"she will",
	"she's":	"she is",
	"should've":	"should have",
	"shouldn't":	"should not",
	"shouldn't've":	"should not have",
	"that'll":	"that will",
	"that're":	"that are",
	"that's":	"that is",
	"that'd":	"that would / that had",
	"there'd":	"there had / there would",
	"there'll":	"there shall / there will",
	"there're":	"there are",
	"there's":	"there is",
	"these're":	"these are",
	"these've":	"these have",
	"they'd":	"they had",
	"they'll":	"they will",
	"they're":	"they are / they were",
	"they've":	"they have",
	"this's":	"this is",
	"those're":	"those are",
	"those've":	"those have",
	"'tis":	"it is",
	"to've":	"to have",
	"'twas":	"it was",
	"wanna":	"want to",
	"wasn't":	"was not",
	"we'd":	"we had / we would/ we did",
	"we'd've":	"we would have",
	"we'll":	"we shall / we will",
	"we're":	"we are",
	"we've":	"we have",
	"weren't":	"were not",
	"what'd":	"what did",
	"what'll":	"what shall / what will",
	"what're":	"what are/what were",
	"what's":	"what is",
	"what've":	"what have",
	"when's":	"when has / when is",
	"where'd":	"where did",
	"where'll":	"where shall / where will",
	"where're":	"where are",
	"where's":	"where has / where is / where does",
	"where've":	"where have",
	"which'd":	"which had / which would",
	"which'll":	"which shall / which will",
	"which're":	"which are",
	"which's":	"which has / which is",
	"which've":	"which have",
	"who'd":	"who would / who had / who did",
	"who'd've":	"who would have",
	"who'll":	"who shall / who will",
	"who're":	"who are",
	"who's":	"who has / who is / who does",
	"who've":	"who have",
	"why'd":	"why did",
	"why're":	"why are",
	"why's":	"why has / why is / why does",
	"willn't":	"will not",
	"won't":	"will not",
	"wonnot":	"will not",
	"would've":	"would have",
	"wouldn't":	"would not",
	"wouldn't've":	"would not have"
	}
	if type(x) is str:
		for key in contractions:
			value = contractions[key]
			x=x.replace(key,value)
		return x
	else:
		return x

def _get_emails(x):
	emails = re.findall(r'([a-z0-9+._-]+@[a-z0-9+._-]+\.[a-z0-9+_-]+\b)',x)
	counts = len(emails)

	return counts, emails


def _remove_emails(x):
	return re.sub(r'([a-z0-9+._-]+@[a-z0-9+._-]+\.[a-z0-9+_-]+)',"",x)


def _get_urls(x):
	urls = re.findall(r'(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?',x)
	counts = len(urls)

	return counts, urls

def _remove_urls(x):
	return re.sub(r'(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', '' , x)


def _remove_rt(x):
	return re.sub(r'\brt\b', '' , x).strip()


def _remove_spacial_chars(x):
	x = re.sub(r'[^\w ]+', "" , x)
	x = ' '.join(x.split())
	return x


def _remove_html_tags(x):
	return BeautifulSoup(x, 'lxml').get_text().split()


def _remove_accented_chars(x):
	x = unicodedata.normalize('NFKD', x).encode('ascii','ignore').decode('utf-8', 'ignore')	

def _remove_stopwords(x):
	return ' '.join([t for t in x.split() if t not in stopwords])	


def _make_base(x):
	x = str(x)
	x_list = []
	doc = nlp(x)

	for token in doc:
		lemma = token.lemma_
		if lemma == '-PRON-' or lemma =='be':
			lemma == token.text

		x_list.append(lemma)
		return ''.join(x_list)

def _remove_common_words(x,n=20):
	text = x.split()
	freq_comm = pd.Series(text).value_counts()
	fn = freq_comm[:n]

	x = ' '.join([t for t in x.split() if t not in fn])
	return x


def _remove_rarewords(x, n=20):
	text = x.split()
	freq_comm = pd.Series(text).value_counts()
	fn = freq_comm.tail(n)	

	x = ' '.join([t for t in x.split() if t not in fn])
	return x


def _spelling_correction(x):
	x = TextBlob(x).correct()
	return x			