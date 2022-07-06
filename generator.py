from General.NLPParser import NLPParser as NLP
import General.funciones as fc

from ComponentExtractors.EntitiesExtractor import EntitiesExtractor as EE
from ComponentExtractors.ClassesExtractor import ClassesExtractor as CE

txtArchivo = 'USTest.txt' #input('Ruta o nombre del archivo de texto: ')


NLPObj = NLP()
EntitiesExtr = EE()
ClassExtr = CE()

with open(txtArchivo) as f_obj:
    lines = f_obj.readlines()

StoryEntities = []

ClassList=[]
ClassRelationsList=[]

ClassRelations={
    "ASOC":{
        "R1":[],
        "R2":[],
        "R3":[],
        "R4":[]
    },
    "HER":{
        "H4":[],
        "H5":[]
    },
    "COMP":{
        "H1":[],
        "H2":[]
    },
    "AGR":{
        "H3":[]
    },
    "DEP":[]
}
numberLine = 1

for line in lines:
    print('Historia #'+str(numberLine)+' ...')
    arrayUSTagged = fc.userStoryTagged(line, NLPObj)

    #Entidades
    StoryEntities = EntitiesExtr.EntitiesStory(arrayUSTagged, StoryEntities)

    #Clases
    ClassList.append(CE.ClassesExtraction(arrayUSTagged))

    ClassRelations = CE.RelationsExtraction(arrayUSTagged,ClassRelations)

    numberLine = numberLine + 1

f_obj.close()

EntitiesList = EntitiesExtr.EntitiesExtraction(StoryEntities)

print(EntitiesList)