import sklearn
import nltk
import numpy as np
import funciones as fc
from NLPParser import NLPParser as NLP
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



class EntitiesExtractor:

    def __init__(self):
        self.RegexParser = NLP.RegexParser()

    def EntitiesStory(self, arrayTagged, StoryEntities):
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
    
    def EntitiesExtraction(self, StoriesEntities):
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(StoriesEntities)
        frecuencias=X.toarray()
        feature_names=vectorizer.get_feature_names()
        frec_proba=(frecuencias.sum(axis=0))/np.sum(frecuencias)
        new_list = [[frec_proba[i], feature_names[i]] for i in range(0, len(frec_proba))]
        new_list.sort(reverse=True, key=lambda x: x[0])

        entity_list = [new_list[i][1] for i in range(0, len(new_list))]

        return entity_list
