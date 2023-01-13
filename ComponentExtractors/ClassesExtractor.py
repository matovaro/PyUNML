import sklearn
import nltk
import numpy as np
import General.funciones as fc
from General.NLPParser import NLPParser as NLP
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

rulesClases = {
    'Clases':{
        'Reglas':[
            'R2: {(<NOUN>|<PROPN>)(<ADJ>|<DET>)*(<NOUN>|<PROPN>)*}'           
            ]
        
    },
    'Relaciones':{
          'Reglas':
            r"""
              R1: {<SUST>(<EXC_TERM.*>)*(<EXC_TERM2>|<EXC_TERM3>)*<VERB><CCONJ>*<VERB>*(<EXC_TERM4>|<EXC_TERM1>)*<SUST>}
              R3:  {(<SUST>)(<EXC_TERM1>)*<ADP>(<EXC_TERM1>)*(<SUST>)}
              H4:  {(<SUST>)(<EXC_TERM4>)*<AUX>(<EXC_TERM4>)*(<SUST>)}
              EXC_TERM4: {(<ADP>|<DET>|<ADJ>)*}
              EXC_TERM1: {(<ADJ>|<DET>)*}
              EXC_TERM2: {<PUNCT><VERB>(<AUX>|<SCONJ>|<PRON>)*}
              EXC_TERM3: {(<PROPN>|<PRON>|<ADV>)*}
              SUST: {(<SUST_1>|<SUST_2>)+}
              SUST_1: {(<NOUN>|<PROPN>)((<EXC_TERM4>)+(<NOUN>|<PROPN>)+)+}
              SUST_2: {(<NOUN>|<PROPN>)+}
              """
            
          ,
          'Excluir' : [
                'EXC_TERM1',
                'EXC_TERM2', 
                'EXC_TERM3', 
                'EXC_TERM4',      
          ],
          'Incluir':[
                "R1",
                "R3",
                "H4"
          ],
          'TO_BE_AUX' :[
                'soy',
                'eres',
                'es',
                'somos',
                'son',
                'sean'        
          ],
          'TYPE_OPTIONS':[
                'tipo',
                'clase',
                'categoría',
                'especie',
                'genero',
                'variedad'
          ],
          'INCLUDE_VERBS' :[
                'comprenden',
                'comprende',
                'consiste',
                'consisten',
                'tiene',
                'tienen',
                'incluyen',
                'incluye',
                'abarcan',
                'abarca',
                'encierran',
                'encierra',
                'engloban',
                'engloba',
                'abrazan',
                'abraza',
                'contienen',
                'contiene'      
          ]
    }
}

classRules = '\n'.join(rulesClases['Clases']['Reglas'])
class ClassesExtractor:
    def __init__(self):
        self.classRegexParser = NLP.RegexParser(classRules)
        self.relationsRegexParser = NLP.RegexParser(rulesClases['Relaciones']['Reglas'], 4)

    def ClassRegexParser(self):
        return self.classRegexParser
    
    def RelationRegexParser(self):
        return self.relationsRegexParser

    def extractClasses(self, arrayTagged):
        classStringArray = []
        classParsedStory = self.classRegexParser.parse(arrayTagged)

        for n in classParsedStory:
            if(type(n)==nltk.tree.Tree):
                word = fc.getNoun(n)
                if fc.verifyClassRules(word):
                    classStringArray.append("_".join(word))
                classString=" ".join(classStringArray)
        
        return classString

    def extractRelations(self, arrayTagged, relations):
        relationParsedStory = self.relationsRegexParser.parse(arrayTagged)

        relaciones = fc.getRelations(relationParsedStory, relations, rulesClases['Relaciones']['Incluir'], rulesClases['Relaciones']['Excluir'], rulesClases)

        return relaciones
    
    def debugRelations(self, arrRel):
        relaciones = []
        entidadesRelaciones = []
        #ASOC
        relacionesAsoc = arrRel['ASOC']['R1'] + arrRel['ASOC']['R2'] + arrRel['ASOC']['R3'] + arrRel['ASOC']['R4']
        arrRelAssigned = []
        for i in relacionesAsoc:
            boolOrderA = i[0]+'#-#'+i[2] in arrRelAssigned
            #boolOrderB = i[2]+'#ASOC#'+i[0] in arrRelAssigned

            if not boolOrderA and i[0] != i[2]:
                relaciones.append([i[0],i[2],'ASOC'])
                arrRelAssigned.append(i[0]+'#-#'+i[2])
                arrRelAssigned.append(i[2]+'#-#'+i[0])

                entidadesRelaciones.append(i[0])
                entidadesRelaciones.append(i[2])

        #HER
        relacionesHer = arrRel['HER']['H4'] + arrRel['HER']['H5']
        for i in relacionesHer:
            boolOrderA = i[0]+'#-#'+i[2] in arrRelAssigned
            #boolOrderB = i[2]+'#ASOC#'+i[0] in arrRelAssigned

            if not boolOrderA and i[0] != i[2]:
                relaciones.append([i[0],i[2],'HER'])
                arrRelAssigned.append(i[0]+'#-#'+i[2])
                arrRelAssigned.append(i[2]+'#-#'+i[0])
                entidadesRelaciones.append(i[0])
                entidadesRelaciones.append(i[2])

        #AGR
        relacionesAgr = arrRel['AGR']['H3']
        for i in relacionesAgr:
            boolOrderA = i[0]+'#-#'+i[2] in arrRelAssigned
            #boolOrderB = i[2]+'#ASOC#'+i[0] in arrRelAssigned

            if not boolOrderA and i[0] != i[2]:
                relaciones.append([i[0],i[2],'AGR'])
                arrRelAssigned.append(i[0]+'#-#'+i[2])
                arrRelAssigned.append(i[2]+'#-#'+i[0])
                entidadesRelaciones.append(i[0])
                entidadesRelaciones.append(i[2])
        entidadesRelaciones = list(set(entidadesRelaciones))
        return relaciones,entidadesRelaciones,arrRelAssigned

    def getCountAparitionsObject(self, objeto, array, index):
        counter = 0
        objetosRelacionados = []

        if index == 0:
            index_rel = 2
        else:
            index_rel = 0
        for i in array:
            if objeto == i[index] and i[index_rel] not in objetosRelacionados:
                counter += 1
                objetosRelacionados.append(i[index_rel])
        
        return counter,objetosRelacionados

    
    def ClassesProcessing(self, ClassRelations, ClassList):
        relacionesProcesadas,preClases,arrClavesRelaciones = self.debugRelations(ClassRelations)
        classes= fc.getResultsFrequency(ClassList)

        ######################################################################
        ######################            ####################################
        ###################### Relaciones ####################################
        ######################            ####################################
        ######################################################################
        relacionesFinales = []

        #Obtiene las "clases" de las relaciones, determina si se encuentran en las clases obtenidas y las guarda en una lista
        clasesRelacionadas = []
        #for rel in relacionesProcesadas:
        for rel in ClassRelations['ASOC']['R3']:
            ## Revisa si el sustantivo de la relacion puede ser reducido y lo hace si si
            pclass2 = rel[2].split('_')[0]
            aparicionesClass2,objRelClass2 = self.getCountAparitionsObject(pclass2, ClassRelations['COMP']['H2'], 0)
            if pclass2 in preClases or aparicionesClass2 > 1:
                rel[2] = pclass2

            
            if rel[2] in classes['Resultados']:
                clasesRelacionadas.append(rel[2])

        newClasesR3 = []
        for classe in classes['Resultados']:
            if classe in clasesRelacionadas:
                newClasesR3.append(classe)
        #Obtiene las "clases" de las relaciones, determina si se encuentran en las clases obtenidas y las guarda en una lista
        clasesRelacionadas = []
        #for rel in relacionesProcesadas:
        for rel in ClassRelations['ASOC']['R1']:
            ## Revisa si el sustantivo de la relacion puede ser reducido y lo hace si si
            pclass1 = rel[0].split('_')[0]
            pclass2 = rel[2].split('_')[0]
            aparicionesClass1,objRelClass1 = self.getCountAparitionsObject(pclass1, ClassRelations['COMP']['H2'], 0)
            aparicionesClass2,objRelClass2 = self.getCountAparitionsObject(pclass2, ClassRelations['COMP']['H2'], 0)
            if pclass1 in preClases or aparicionesClass1 > 1:
                rel[0] = pclass1
            if pclass2 in preClases or aparicionesClass2 > 1:
                rel[2] = pclass2

            
            if rel[0] in classes['Resultados'] and rel[2] in classes['Resultados']:
                #relacionesFinales.append(rel)
                clasesRelacionadas.append(rel[0])
                clasesRelacionadas.append(rel[2])
            else:
                if rel[0] in classes['Resultados']:
                    clasesRelacionadas.append(rel[0])
                if rel[2] in classes['Resultados']:
                    clasesRelacionadas.append(rel[2])

        #recorre la lsita obtenida anteriormente y elimina las clases que no se encuentren en ella
        newClasesR1 = []
        for classe in classes['Resultados']:
            if classe in clasesRelacionadas:
                newClasesR1.append(classe)

        classes['Resultados'] = list(set(newClasesR1+newClasesR3))


        for rel in relacionesProcesadas:
            ## Revisa si el sustantivo de la relacion puede ser reducido y lo hace si si
            pclass1 = rel[0].split('_')[0]
            pclass2 = rel[1].split('_')[0]
            aparicionesClass1,objRelClass1 = self.getCountAparitionsObject(pclass1, ClassRelations['COMP']['H2'], 0)
            aparicionesClass2,objRelClass2 = self.getCountAparitionsObject(pclass2, ClassRelations['COMP']['H2'], 0)
            if pclass1 in preClases or aparicionesClass1 > 1:
                rel[0] = pclass1
            if pclass2 in preClases or aparicionesClass2 > 1:
                rel[1] = pclass2

            if rel[0] in classes['Resultados'] and rel[1] in classes['Resultados'] and rel[0] != rel[1]:
                relacionesFinales.append(rel)

        ######################################################################
        ######################                      ##########################
        ######################       Atributos      ##########################
        ######################                      ##########################
        ######################################################################

        atributos = []
        for r in ClassRelations['COMP']['H1']:
            atributos.append((r[0], r[2]))

        for r in ClassRelations['COMP']['H2']:
            atributos.append((r[2], r[0]))

        ######################################################################
        ######################                      ##########################
        ######################        Metodos       ##########################
        ######################                      ##########################
        ######################################################################
        metodos = []

        for rel in ClassRelations['ASOC']['R1']:
            ## Revisa si el sustantivo de la relacion puede ser reducido y lo hace si si
            pclass1 = rel[0].split('_')[0]
            pclass2 = rel[2].split('_')[0]
            if pclass1 in classes['Resultados']:
                rel[0] = pclass1
            if pclass2 in classes['Resultados']:
                rel[2] = pclass2

            
            if rel[0] in classes['Resultados']:
                #relacionesFinales.append(rel)
                metodos.append((rel[0],'_'.join([rel[1],rel[2]])))

        arregloClases = fc.getClassesArray(atributos, metodos, relacionesFinales)
        classes['Resultados'] = arregloClases['Clases'].keys()

        return arregloClases
