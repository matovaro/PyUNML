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
    def _init_(self):

        self.spacy = spacy.load('es_core_news_lg')
        self.stanza = stanza.Pipeline(lang='es', processors='tokenize,mwt,pos,lemma,depparse,ner',package='ancora')

    def SpacyParser(self):
        return self.spacy

    def StanzaParser(self):
        return self.stanza

    def RegexParser():
        return nltk.RegexpParser(grammarEspanish)