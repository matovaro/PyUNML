

class DiagramFileGenerator:

    def __init__(self, FileName):
        fileName = FileName.split('/')
        fileName = fileName[len(fileName)-1].split('\\')
        fileName = fileName[len(fileName)-1].split('.')[0]
        self.FileName = fileName

    def EntitiesFile(self, entities):
        nameFile = self.FileName + '-Entities.txt'
        EntFile = open(nameFile,'a')

        EntFile.write('---------- LISTA DE ENTIDADES -----------' +'\n')
        EntFile.write('* A continuación encontrara la lista de entidades presentes en las historias de usuario. Estan organizadas de la entidad mas frecuente a la menos frecuente *' +'\n')

        for entity in entities:
            EntFile.write(entity +'\n')
        
        EntFile.close()
        
        return nameFile