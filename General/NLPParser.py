import spacy
import stanza
import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.chunk import conlltags2tree, tree2conlltags 
from nltk.chunk import ne_chunk

class NLPParser:
    def __init__(self):
        self.sp = spacy.load('es_core_news_lg')
        self.st = stanza.Pipeline(lang='es', processors='tokenize,mwt,pos,lemma,depparse,ner',package='ancora')

    def SpacyParser(self):
        return self.sp

    def StanzaParser(self):
        return self.st

    def RegexParser(rules, loops = 1):
        return nltk.RegexpParser(rules, loop=loops)