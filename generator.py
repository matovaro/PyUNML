from NLPParser import NLPParser as NLP
import funciones as fc

from ComponentExtractors.EntitiesExtractor import EntitiesExtractor as EE

txtArchivo = 'USTest.txt' #input('Ruta o nombre del archivo de texto: ')


NLPObj = NLP()
EntitiesExtr = EE()

with open(txtArchivo) as f_obj:
    lines = f_obj.readlines()

StoryEntities = []
numberLine = 1

for line in lines:
    print('Historia #'+str(numberLine)+' ...')
    arrayUSTagged = fc.userStoryTagged(line, NLPObj)
    StoryEntities = EntitiesExtr.EntitiesStory(arrayUSTagged, StoryEntities)


    numberLine = numberLine + 1

f_obj.close()

EntitiesList = EntitiesExtr.EntitiesExtraction(StoryEntities)

print(EntitiesList)