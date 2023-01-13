from General.NLPParser import NLPParser as NLP
import General.funciones as fc

from ComponentExtractors.EntitiesExtractor import EntitiesExtractor as EE
from ComponentExtractors.ClassesExtractor import ClassesExtractor as CE
from ComponentExtractors.CaseUseExtractor import CaseUseExtractor as CUE
from Generators.DiagramFileGenerator import DiagramFileGenerator as DFG

txtArchivo = input('Ruta o nombre del archivo de texto: ')


NLPObj = NLP()
EntitiesExtr = EE()
ClassExtr = CE()
CaseUseExtr = CUE()
FileGenerator = DFG(txtArchivo)

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
    arrayUSTagged = fc.tagUserStory(line, NLPObj)
    arrayUSTaggedShort = fc.tagUserStory(line.split(',')[0], NLPObj)

    #Entidades
    EntitiesList = EntitiesExtr.getStoryEntities(arrayUSTagged, EntitiesList)

    #Clases
    ClassList.append(ClassExtr.extractClasses(arrayUSTagged))

    ClassRelations = ClassExtr.extractRelations(arrayUSTagged, ClassRelations)

    #Casos de uso
    ActorsList, ActorsRelations = CaseUseExtr.extractCaseUse(arrayUSTagged, arrayUSTaggedShort, ActorsList, ActorsRelations)

    numberLine = numberLine + 1

f_obj.close()

StoryEntities = EntitiesExtr.extractEntities(EntitiesList)

StoryClasses = ClassExtr.ClassesProcessing(ClassRelations, ClassList)

StoryCaseUse = CaseUseExtr.CaseUseProcessing(ActorsList, ActorsRelations)

print('')
print('############################# ENTIDADES ###################################')
#print(StoryEntities)
entFile = FileGenerator.getEntitiesFile(StoryEntities)
print(entFile)

print('')
print('############################# CLASES ###################################')
#print(StoryClasses)
classFile = FileGenerator.getClassFile(StoryClasses)
print(classFile)

print('')
print('############################# CASOS DE USO ###################################')
#print(StoryCaseUse)
caseFile = FileGenerator.getCaseUseFile(StoryCaseUse)
print(caseFile)