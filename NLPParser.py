import spacy
import stanza
import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.chunk import conlltags2tree, tree2conlltags 
from nltk.chunk import ne_chunk

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('stopwords')

stanza.download('es') # download English model
stanza.download(lang="es",package=None,processors={"ner":"ancora"})

grammarEspanish = '\n'.join([
  'R2: {(<NOUN>|<PROPN>)<ADJ><DET>?(<NOUN>|<PROPN>)*}',
  'R3: {(<NOUN>|<PROPN>)}'
  
	])

class NLPParser:
    def __init__(self):
        self.sp = spacy.load('es_core_news_lg')
        self.st = stanza.Pipeline(lang='es', processors='tokenize,mwt,pos,lemma,depparse,ner',package='ancora')

    def SpacyParser(self):
        return self.sp

    def StanzaParser(self):
        return self.st

    def RegexParser():
        return nltk.RegexpParser(grammarEspanish)