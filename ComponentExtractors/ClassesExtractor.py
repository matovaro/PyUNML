import sklearn
import nltk
import numpy as np
import General.funciones as fc
from General.NLPParser import NLPParser as NLP
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

rules = {
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
                'categoria',
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

classRules = '\n'.join(rules['Clases']['Reglas'])
class ClassesExtractor:
    def __init__(self):
        self.classRegexParser = NLP.RegexParser(classRules)
        self.relationsRegexParser = NLP.RegexParser(rules['Relaciones']['Reglas'], 4)

    def ClassRegexParser(self):
        return self.classRegexParser
    
    def RelationRegexParser(self):
        return self.relationsRegexParser

    ########################### Funciones

    # Metodo para la remoción de palabras que no seran tenidas en cuenta en el analisis, debido a que podrian causar ruido
    def verificationReglasClase(self, word):
        
        #Pendiente: Ajustar resultados de patrones a condiciones de evitar estas palabras segun Btoush y la relacion "A is a B"
        J= ['número', 'no', 'codigo', 'fecha', 'tipo', 'volumen', 'nacimiento', 'id', 'dirección', 'nombre']
        stop_sustantivo=['base_de_datos','base_de_dato','base_dato', 'registro', 'sistema', 'información', 'organización',  'detalle','cosa']

        
        totalWord = '_'.join(word)
        boolTotalWord = True
        if (totalWord in J) or (totalWord in stop_sustantivo):
            boolTotalWord = False
        comparacionJ = [item for item in word if item in J]
        comparacionStop = [item for item in word if item in stop_sustantivo]

        if (len(comparacionJ) <=0 and len(comparacionStop) <=0 and boolTotalWord ):
            return True
        else:
            return False

    #Metodo que determina si una palabra esta en plural o singular, devuelve su correspondiente singular y retorna un array con las palabras del sustantivo compuesto
    def contructionWordSustantivo(self, ntree, exclusionRules = []):
        #print(n.label())
        word=[]
        word2=[]
        for i in ntree:

            if(type(i)==nltk.tree.Tree and i.label() not in exclusionRules):
                wordTemp = self.contructionWordSustantivo(i, exclusionRules)
                if wordTemp:
                    word.append(wordTemp)
            elif(type(i)!=nltk.tree.Tree):
                caracteristicas = i[4][0][0].split('|')

                if i[1] == 'NOUN' or i[1] == 'PROPN':
                    if 'Number=Plur' in caracteristicas:
                        word.append(i[3])
                    elif 'Number=Sing' in caracteristicas:
                        word.append(i[0])
                    else:
                        word.append(i[0])
            #word2.append([i[0],i[1],i[2],i[3],i[4]])
        return word

    def obtenerRelacionesSustComp(self, arrSustantivo, unionWord = 'UNION'):
        arrRelaciones = []
        idx_1 = 0
        idx_2 = 1
        while idx_2 <= len(arrSustantivo) - 1:
            arrRelaciones.append([arrSustantivo[idx_1],unionWord,arrSustantivo[idx_2]])
            idx_1 += 1
            idx_2 += 1
        return arrRelaciones

    def relacionesSustComp(self, txtRegla,arrDatos):
        arrRelaciones = []
        if(txtRegla == "H2"):
            # Particionamos el sust. compuesto en sustantivos simples
            arrSustantivo = arrDatos.split('_')
            if len(arrSustantivo) > 1:
                # Si hay mas de un sustantivo, creamos una relacion por cada pareja consecutiva del 
                # sust. compuesto
                arrRelaciones = self.obtenerRelacionesSustComp(arrSustantivo)
        elif(txtRegla == "R3"):
            txtADP = ''
            susts = [[]]
            # Recorremos el arreglo de tokens y POS para determinar los sustantivos unidos por un ADP 
            # y guardamos cada conjunto de sustantivos separado por un ADP en un arreglo distinto
            for ele in arrDatos:
                if ele[1]=='NOUN' or ele[1]=='PROPN':
                    susts[len(susts)-1].append(ele[0])
                elif ele[1] == 'ADP':
                    susts.append([])
            
            # Si hay mas de un conjunto detectado, concatenamos cada conjunto de sustantivos
            # y creamos una relacion por cada pareja consecutiva
            if len(susts) > 1:
                arrTemp = []
                for itm in susts:
                    arrTemp.append('_'.join(itm))
                
                arrRelaciones = self.obtenerRelacionesSustComp(arrTemp,'ADP')

        return arrRelaciones


    def relacionesHerencia(self, arrRelacion, arrRelacionesTotal):
        sustComp = arrRelacion[2].split('_')
        if(sustComp[0] in rules['Relaciones']['TYPE_OPTIONS'] and len(sustComp) > 1):
            newSust = '_'.join(sustComp[1:])
            arrRelacion[2] = newSust
            arrRelacionesTotal["HER"]["H5"].append(arrRelacion)
        else:
            arrRelacionesTotal["HER"]["H4"].append(arrRelacion)
        return arrRelacionesTotal

    def flatTree(self, parsed,arrItems):
        for ele in parsed:
            if isinstance(ele, list):
                arrItems = self.flatTree(ele,arrItems)
            else:
                arrItems.append((ele[0], ele[1]))
        return arrItems

    def obtenerRelaciones(self, arrayParsed,relaciones,inclusionRules = [],exclusionRules = []):
  
        for ele in arrayParsed:
            word=[]
            #Validamos si el elemento es un arbol
            if(type(ele)==nltk.tree.Tree):

                #Validamos si el elemento esta entre las reglas aceptadas
                if ele.label() in inclusionRules:
                    relacion = []

                    #Recorremos los elementos de la regla y construimos la relacion
                    for item in ele:
                        if type(item)==nltk.tree.Tree:
                            if item.label() == 'SUST':
                                txtSustantivo = '_'.join(self.contructionWordSustantivo(item,exclusionRules)[0])
                                relaciones["COMP"]["H2"] = relaciones["COMP"]["H2"] + self.relacionesSustComp("H2",txtSustantivo)
                                arrTreeFlat = self.flatTree(item,[])
                                relaciones["ASOC"]["R3"] = relaciones["ASOC"]["R3"] + self.relacionesSustComp("R3",arrTreeFlat)
                                relacion.append(txtSustantivo)
                        elif item[1]!='CCONJ':
                            if ele.label() in ["R1","R3"]:
                                # COMP_H1
                                if(item[0] in rules['Relaciones']['INCLUDE_VERBS']):
                                    relacion.append(item[0])
                                else:
                                    relacion.append(item[3])
                            elif ele.label() in ["H4"]:
                                if(item[0] in rules['Relaciones']['TO_BE_AUX']):
                                    relacion.append(item[3])

                    #Validamos si la relacion contiene mas de una union y crea multiples relaciones en caso de que asi sea
                    if len(relacion) > 3:
                        for index in range(1,len(relacion)-1):
                            
                            if ele.label() in ["R1","R3"]:
                                relacionTemp = [relacion[0],relacion[index],relacion[len(relacion)-1]]
                                if relacion[index] in rules['Relaciones']['INCLUDE_VERBS']:
                                    relaciones["COMP"]["H1"].append(relacionTemp)
                                else:
                                    relaciones["ASOC"][ele.label()].append(relacionTemp)
                            elif ele.label() in ["H4"]:
                                # H4 y H5
                                relacionTemp = [relacion[0],relacion[index],relacion[len(relacion)-1]]
                                relaciones = self.relacionesHerencia(relacionTemp,relaciones)
                                #relaciones["HER"]["H4"].append(relacionTemp)
                    elif len(relacion) == 3:
                        #Guarda la relacion en la categoria correspondiente
                        if ele.label() in ["R1","R3"]:
                            # COMP_H1
                            if relacion[1] in rules['Relaciones']['INCLUDE_VERBS']:
                                relaciones["COMP"]["H1"].append(relacion)
                            else:
                                relaciones["ASOC"][ele.label()].append(relacion)
                        elif ele.label() in ["H4"]:
                            # H4 y H5
                            relaciones = self.relacionesHerencia(relacion,relaciones)
                            #relaciones["HER"]["H4"].append(relacion)
        return relaciones

    ## Analiza un string recibido y retorna la frecuencia de los terminos que incluye
    def obtenerResultadosFrecuencia(self, txtStringComponentes):
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(txtStringComponentes)
        frecuencias=X.toarray()
        feature_names=vectorizer.get_feature_names_out()
        frec_proba=(frecuencias.sum(axis=0))/np.sum(frecuencias)
        new_list = [[frec_proba[i], feature_names[i]] for i in range(0, len(frec_proba))]
        new_list.sort(reverse=True, key=lambda x: x[0])
        component_list = [new_list[i][1] for i in range(0, len(new_list))]
        componentsFrecuencyStory = new_list
        componentsStory = component_list
        return {'Frecuencia': componentsFrecuencyStory, 'Resultados':componentsStory}

    def ClassesExtraction(self, arrayTagged):
        classStringArray = []
        classParsedStory = self.classRegexParser.parse(arrayTagged)

        for n in classParsedStory:
            word=[]
            if(type(n)==nltk.tree.Tree):
                word = self.contructionWordSustantivo(n)
                if self.verificationReglasClase(word):
                    classStringArray.append("_".join(word))
                classString=" ".join(classStringArray)
        
        return classString

    def RelationsExtraction(self, arrayTagged,relations):

        relationParsedStory = self.relationsRegexParser.parse(arrayTagged)

        test = []
        for pr in arrayTagged:
            test.append((pr[0],pr[1]))
        relationTester = self.relationsRegexParser.parse(test)
        #print(relationTester)
        
        relaciones = self.obtenerRelaciones(relationParsedStory,relations,rules['Relaciones']['Incluir'],rules['Relaciones']['Excluir'])
        #print(relaciones)
        
        return relaciones
    
    def depuracionRelaciones(self, arrRel):
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

    def numAparObjeto(self, objeto,array,index):
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

    def construirArregloClases(self, atributos,metodos,relaciones):
        arregloDiagrama = {}
        clases = {}
        for atributo in atributos:
            if atributo[0] not in clases:
                clases[atributo[0]] = {}
                clases[atributo[0]]['Atributos'] = []
                clases[atributo[0]]['Metodos'] = []
            
            if atributo[1] not in clases[atributo[0]]['Atributos']:
                clases[atributo[0]]['Atributos'].append(atributo[1])
        
        for metodo in metodos:
            if metodo[0] not in clases:
                clases[metodo[0]] = {}
                clases[metodo[0]]['Atributos'] = []
                clases[metodo[0]]['Metodos'] = []
            
            if metodo[1] not in clases[metodo[0]]['Metodos']:
                clases[metodo[0]]['Metodos'].append(metodo[1])
        
        arregloDiagrama['Clases'] = clases
        arregloDiagrama['Relaciones'] = []

        relacionesAdicionadas = []
        for relacion in relaciones:
            if relacion[0] in clases.keys() and relacion[1] in clases.keys() and [relacion[0],relacion[1]] not in relacionesAdicionadas:
                arregloDiagrama['Relaciones'].append(relacion)
                relacionesAdicionadas.append([relacion[0],relacion[1]])
                relacionesAdicionadas.append([relacion[1],relacion[0]])
        
        return arregloDiagrama

    def ClassesProcessing(self, ClassRelations, ClassList):
        relacionesProcesadas,preClases,arrClavesRelaciones = self.depuracionRelaciones(ClassRelations)
        classes= self.obtenerResultadosFrecuencia(ClassList)

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
            aparicionesClass2,objRelClass2 = self.numAparObjeto(pclass2,ClassRelations['COMP']['H2'],0)
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
            aparicionesClass1,objRelClass1 = self.numAparObjeto(pclass1,ClassRelations['COMP']['H2'],0)
            aparicionesClass2,objRelClass2 = self.numAparObjeto(pclass2,ClassRelations['COMP']['H2'],0)
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
            aparicionesClass1,objRelClass1 = self.numAparObjeto(pclass1,ClassRelations['COMP']['H2'],0)
            aparicionesClass2,objRelClass2 = self.numAparObjeto(pclass2,ClassRelations['COMP']['H2'],0)
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


        preAtributos = ClassRelations['COMP']['H1'] + ClassRelations['COMP']['H2']
        atributos = []
        for r in ClassRelations['COMP']['H1']:
            aparicionesH1,objRelH1 = self.numAparObjeto(r[2],ClassRelations['COMP']['H1'],2) 
            aparicionesH2,objRelH2 = self.numAparObjeto(r[2],ClassRelations['COMP']['H2'],0) 
            boolUnicaRelacion = aparicionesH1 < 2 and aparicionesH2 < 2 and len(list(set(objRelH1 + objRelH2))) < 2

            if r[2] not in classes['Resultados']:
                atributos.append((r[0],r[2]))

        for r in ClassRelations['COMP']['H2']:
            aparicionesH1,objRelH1 = self.numAparObjeto(r[0],ClassRelations['COMP']['H1'],2) 
            aparicionesH2,objRelH2 = self.numAparObjeto(r[0],ClassRelations['COMP']['H2'],0) 
            boolUnicaRelacion = aparicionesH1 < 2 and aparicionesH2 < 2 and len(list(set(objRelH1 + objRelH2))) < 2

            if r[0] not in classes['Resultados']:
                atributos.append((r[2],r[0]))

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

        arregloClases = self.construirArregloClases(atributos,metodos,relacionesFinales)
        classes['Resultados'] = arregloClases['Clases'].keys()

        return arregloClases