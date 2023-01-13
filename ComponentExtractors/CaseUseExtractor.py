import sklearn
import nltk
import numpy as np
import General.funciones as fc
from General.NLPParser import NLPParser as NLP
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

rulesActors = {
    'Actores':{
        'Reglas':
            r"""
                EXC_TERM1: {(<ADP>|<DET>|<ADJ>)*}
                SUST: {(<SUST_1>|<SUST_2>)+}
                SUST_1: {(<NOUN>|<PROPN>)((<EXC_TERM1>)+(<NOUN>|<PROPN>)+)+}
                SUST_2: {(<NOUN>|<PROPN>)+}
            """
    },
    'Relaciones':{
          'Reglas':
            r"""
              R1: {<SUST>(<EXC_TERM.*>)*(<EXC_TERM2>|<EXC_TERM3>)*<VERB><CCONJ>*<VERB>*(<EXC_TERM4>|<EXC_TERM1>)*<SUST>}
              EXC_TERM4: {(<ADP>|<DET>|<ADJ>)*}
              EXC_TERM1: {(<ADJ>|<DET>)*}
              EXC_TERM2: {<PUNCT><VERB>(<AUX>|<SCONJ>|<PRON>)*}
              EXC_TERM3: {(<PROPN>|<PRON>|<ADV>)*}
              SUST: {(<SUST_1>|<SUST_2>)+}
              SUST_1: {(<NOUN>|<PROPN>)((<EXC_TERM4>)+(<NOUN>|<PROPN>)+)+}
              SUST_2: {(<NOUN>|<PROPN>)+}
            """
            
          
    }
}

class CaseUseExtractor:

    def __init__(self):
        self.actorsRegexParser = NLP.RegexParser(rulesActors['Actores']['Reglas'], 2)
        self.relationsRegexParser = NLP.RegexParser(rulesActors['Relaciones']['Reglas'], 3)

    def ActorsRegexParser(self):
        return self.actorsRegexParser
    
    def RelationRegexParser(self):
        return self.relationsRegexParser

    def extractCaseUse(self, arrayUSTagged, arrayUSTaggedShort, actors, ActorsRelations):
        countSust = 0

        ###### Extraccion de actor
        actoresParsedStory = self.actorsRegexParser.parse(arrayUSTaggedShort)
        for n in actoresParsedStory:
            if(type(n)==nltk.tree.Tree and n.label() == 'SUST' and countSust == 0):
                countSust += 1
                word = fc.getNoun(n[0])
                actor = "_".join(word)

                if actor not in actors:
                    actors.append(actor)
        
        ###### Extraccion de casos de uso
        relacionesParsedStory = self.relationsRegexParser.parse(arrayUSTagged)
        for n in relacionesParsedStory:
            relacion = []
            if(type(n)==nltk.tree.Tree and n.label() == 'R1'):
                for item in n:
                    if(type(item)==nltk.tree.Tree and item.label() == 'SUST'):
                        word = fc.getNoun(item[0])
                        SUST = "_".join(word)

                        relacion.append(SUST)
                    
                    elif(type(item)!=nltk.tree.Tree and item[1] != 'CCONJ'):
                        relacion.append(item[3])
        
            if len(relacion) > 3:
                for index in range(1,len(relacion)-1):
                
                    relacionTemp = [relacion[0],relacion[index],relacion[len(relacion)-1]]
                    ActorsRelations.append(relacionTemp)
            elif len(relacion) == 3:
                ActorsRelations.append(relacion)
        actors = list(set(actors))
        return actors, ActorsRelations

    def getCaseUseArray(self, arrayCU):
        UseCases = {}

        for cu in arrayCU:
            if cu[0] not in UseCases:
                UseCases[cu[0]] = []
            
            UseCases[cu[0]].append(cu[1])
            arrTemp = list(set(UseCases[cu[0]]))
            UseCases[cu[0]] = arrTemp

        return UseCases

    def CaseUseProcessing(self, actors, ActorsRelations):
        relacionesFinales = []
        for rel in ActorsRelations:
            if rel[0] in actors:
                relacionesFinales.append([rel[0], rel[1]+' '+rel[2]])
        
        UseCases = self.getCaseUseArray(relacionesFinales)
        return UseCases
