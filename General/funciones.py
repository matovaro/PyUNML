import stanza
import nltk
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

def verifyRules(word):
  
  J= ['número', 'no', 'código', 'fecha', 'tipo', 'volumen', 'nacimiento', 'id', 'dirección', 'nombre']
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

def getWord(ntree):
  word=[]
  for i in ntree:
    caracteristicas = i[4][0][0].split('|')

    if 'Number=Plur' in caracteristicas:
      word.append(i[3])
    elif 'Number=Sing' in caracteristicas:
      word.append(i[0])
    #word2.append([i[0],i[1],i[2],i[3],i[4]])
  return word
  
def tagUserStory(txtUserStory, NLPObj):
    arrayWordsTagged = []
    StanzaParser = NLPObj.StanzaParser()
    docStanzaSpanish = StanzaParser(txtUserStory.lower())
    for sent in docStanzaSpanish.sentences: 
        for word in sent.words:
        #print((word.text, word.upos, word.deprel,word.lemma,[[word.feats if word.feats else "_"]]))
            arrayWordsTagged.append((word.text, word.upos, word.deprel,word.lemma,[[word.feats if word.feats else "_"]]))
    
    return arrayWordsTagged

# Metodo para la remoción de palabras que no seran tenidas en cuenta en el analisis, debido a que podrian causar ruido
def verifyClassRules(word):
    
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
def getNoun(ntree, exclusionRules = []):
  word=[]
  for i in ntree:

      if(type(i)==nltk.tree.Tree and i.label() not in exclusionRules):
          wordTemp = getNoun(i, exclusionRules)
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

def getCompNounRelations(arrSustantivo, unionWord ='UNION'):
  arrRelaciones = []
  idx_1 = 0
  idx_2 = 1
  while idx_2 <= len(arrSustantivo) - 1:
      arrRelaciones.append([arrSustantivo[idx_1],unionWord,arrSustantivo[idx_2]])
      idx_1 += 1
      idx_2 += 1
  return arrRelaciones

def getCompNounRelationsByRule(txtRegla, arrDatos):
  arrRelaciones = []
  if(txtRegla == "H2"):
      # Particionamos el sust. compuesto en sustantivos simples
      arrSustantivo = arrDatos.split('_')
      if len(arrSustantivo) > 1:
          # Si hay mas de un sustantivo, creamos una relacion por cada pareja consecutiva del 
          # sust. compuesto
          arrRelaciones = getCompNounRelations(arrSustantivo)
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
          
          arrRelaciones = getCompNounRelations(arrTemp, 'ADP')

  return arrRelaciones

def getInheritanceRelations(arrRelacion, arrRelacionesTotal, rules):
  sustComp = arrRelacion[2].split('_')
  if(sustComp[0] in rules['Relaciones']['TYPE_OPTIONS'] and len(sustComp) > 1):
      newSust = '_'.join(sustComp[1:])
      arrRelacion[2] = newSust
      arrRelacionesTotal["HER"]["H5"].append(arrRelacion)
  else:
      arrRelacionesTotal["HER"]["H4"].append(arrRelacion)
  return arrRelacionesTotal

def getFlatTree(parsed, arrItems):
  for ele in parsed:
      if isinstance(ele, list):
          arrItems = getFlatTree(ele, arrItems)
      else:
          arrItems.append((ele[0], ele[1]))
  return arrItems

def getRelations(arrayParsed, relaciones, inclusionRules = [], exclusionRules = [], rules = []):
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
              txtSustantivo = '_'.join(getNoun(item, exclusionRules)[0])
              relaciones["COMP"]["H2"] = relaciones["COMP"]["H2"] + getCompNounRelationsByRule("H2", txtSustantivo)
              arrTreeFlat = getFlatTree(item, [])
              relaciones["ASOC"]["R3"] = relaciones["ASOC"]["R3"] + getCompNounRelationsByRule("R3", arrTreeFlat)
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
              relaciones = getInheritanceRelations(relacionTemp, relaciones, rules)
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
            relaciones = getInheritanceRelations(relacion, relaciones, rules)
            #relaciones["HER"]["H4"].append(relacion)
  return relaciones

## Analiza un string recibido y retorna la frecuencia de los terminos que incluye
def getResultsFrequency(txtStringComponentes):
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

def getClassesArray(atributos, metodos, relaciones):
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

  clasesFinales = []
  for relacion in relaciones:
      ### Descomentar para mantener clases relacionadas a otras sin metodos o atributos ###
      '''
      if relacion[0] not in clases and relacion[1] in arregloDiagrama['Clases']:
          clases[relacion[0]] = {}
      if relacion[1] not in clases and relacion[0] in arregloDiagrama['Clases']:
          clases[relacion[1]] = {}
      '''
      #####################

      if relacion[0] in clases.keys() and relacion[1] in clases.keys() and [relacion[0],relacion[1]] not in relacionesAdicionadas:
          arregloDiagrama['Relaciones'].append(relacion)
          relacionesAdicionadas.append([relacion[0],relacion[1]])
          relacionesAdicionadas.append([relacion[1],relacion[0]])
          clasesFinales.append(relacion[0])
          clasesFinales.append(relacion[1])
  
  for item in clases.copy().keys():
      if item not in clasesFinales:
          clases.pop(item)

  arregloDiagrama['Clases'] = clases

  return arregloDiagrama
