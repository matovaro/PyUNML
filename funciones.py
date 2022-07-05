from NLPParser import NLPParser as NLP

def verificationReglas(word):
  
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

def contructionWord(ntree):
  #print(n.label())
  word=[]
  word2=[]
  for i in ntree:
    caracteristicas = i[4][0][0].split('|')

    if 'Number=Plur' in caracteristicas:
      word.append(i[3])
    elif 'Number=Sing' in caracteristicas:
      word.append(i[0])
    #word2.append([i[0],i[1],i[2],i[3],i[4]])
  return word


def userStoryTagged(txtUserStory):
  arrayWordsTagged = []
  docStanzaSpanish = NLP.StanzaParser(txtUserStory.lower())
  for sent in docStanzaSpanish.sentences: 
    for word in sent.words:
      #print((word.text, word.upos, word.deprel,word.lemma,[[word.feats if word.feats else "_"]]))
      arrayWordsTagged.append((word.text, word.upos, word.deprel,word.lemma,[[word.feats if word.feats else "_"]]))
  
  return arrayWordsTagged