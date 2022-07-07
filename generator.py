from General.NLPParser import NLPParser as NLP
import General.funciones as fc

from ComponentExtractors.EntitiesExtractor import EntitiesExtractor as EE
from ComponentExtractors.ClassesExtractor import ClassesExtractor as CE
from ComponentExtractors.CaseUseExtractor import CaseUseExtractor as CUE

txtArchivo = 'USTest.txt' #input('Ruta o nombre del archivo de texto: ')


NLPObj = NLP()
EntitiesExtr = EE()
ClassExtr = CE()
CaseUseExtr = CUE()

with open(txtArchivo) as f_obj:
    lines = f_obj.readlines()

EntitiesList = []

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

ActorsList=[]

ActorsRelations = []

numberLine = 1

for line in lines:
    print('Historia #'+str(numberLine)+' ...')
    arrayUSTagged = fc.userStoryTagged(line, NLPObj)
    arrayUSTaggedShort = fc.userStoryTagged(line.split(',')[0], NLPObj)

    #Entidades
    EntitiesList = EntitiesExtr.EntitiesStory(arrayUSTagged, EntitiesList)

    #Clases
    ClassList.append(ClassExtr.ClassesExtraction(arrayUSTagged))

    ClassRelations = ClassExtr.RelationsExtraction(arrayUSTagged,ClassRelations)

    #Casos de uso
    ActorsList, ActorsRelations = CaseUseExtr.CaseUseExtraction(arrayUSTagged, arrayUSTaggedShort, ActorsList, ActorsRelations)

    numberLine = numberLine + 1

f_obj.close()

StoryEntities = EntitiesExtr.EntitiesExtraction(EntitiesList)

StoryClasses = ClassExtr.ClassesProcessing(ClassRelations, ClassList)

StoryCaseUse = CaseUseExtr.CaseUseProcessing(ActorsList, ActorsRelations)

print('')
print('############################# ENTIDADES ###################################')
print(EntitiesList)

print('')
print('############################# CLASES ###################################')
print(StoryClasses)

print('')
print('############################# CASOS DE USO ###################################')
print(StoryCaseUse)