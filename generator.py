from NLPParser import NLPParser as NLP
import funciones as fc

from ComponentExtractors.EntitiesExtractor import EntitiesExtractor as EE

txtArchivo = input('Ruta o nombre del archivo de texto: ')

with open(txtArchivo) as f_obj:
    lines = f_obj.readlines()

StoryEntities = []
for line in lines:

    arrayUSTagged = fc.userStoryTagged(line)

    StoryEntities = EE.EntitiesStory(arrayUSTagged, StoryEntities)