import sklearn
import nltk
import numpy as np
import General.funciones as fc
from General.NLPParser import NLPParser as NLP
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

entitiesRules = '\n'.join([
    'R2: {(<NOUN>|<PROPN>)<ADJ><DET>?(<NOUN>|<PROPN>)*}',
    'R3: {(<NOUN>|<PROPN>)}'
    
    ])

class EntitiesExtractor:

    def __init__(self):
        self.RegexParser = NLP.RegexParser(entitiesRules)

    def getStoryEntities(self, arrayTagged, StoryEntities):
        cs = self.RegexParser.parse(arrayTagged)
        entitiesStringArray=[]
        for n in cs:
          if(type(n)==nltk.tree.Tree):
            word = fc.getWord(n)
            if fc.verifyRules(word):
              entitiesStringArray.append("_".join(word))
            entitiesString=" ".join(entitiesStringArray)
              
        StoryEntities.append(entitiesString)

        return StoryEntities
    
    def extractEntities(self, StoriesEntities):
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(StoriesEntities)
        frecuencias=X.toarray()
        feature_names=vectorizer.get_feature_names_out()
        frec_proba=(frecuencias.sum(axis=0))/np.sum(frecuencias)
        new_list = [[frec_proba[i], feature_names[i]] for i in range(0, len(frec_proba))]
        new_list.sort(reverse=True, key=lambda x: x[0])

        entity_list = [new_list[i][1] for i in range(0, len(new_list))]

        return entity_list
