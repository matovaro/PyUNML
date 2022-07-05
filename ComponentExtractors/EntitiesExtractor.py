import sklearn
import nltk
import funciones as fc
from NLPParser import NLPParser as NLP
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



class EntitiesExtractor:

    def _init_(self):
        self.RegexParser = NLP.RegexParser()

    def EntitiesStory(self, arrayTagged,StoryEntities):
        cs = self.RegexParser.parse(arrayTagged)
        entitiesStringArray=[]
        for n in cs:
          word=[]
          if(type(n)==nltk.tree.Tree):
            word = fc.contructionWord(n)
            if fc.verificationReglas(word):
              entitiesStringArray.append("_".join(word))
            entitiesString=" ".join(entitiesStringArray)
              
        StoryEntities.append(entitiesString)

        return StoryEntities
