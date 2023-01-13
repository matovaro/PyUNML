from os import system, mkdir
import errno

class DiagramFileGenerator:

    def __init__(self, FileName):
        fileName = FileName.split('/')
        fileName = fileName[len(fileName)-1].split('\\')
        fileName = fileName[len(fileName)-1].split('.')[0]
        self.FileName = fileName
        self.ResultsFolder = 'Resultados'

        try:
            mkdir(self.ResultsFolder)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


    def getEntitiesFile(self, entities):
        nameFile = self.ResultsFolder + '/' + self.FileName + '-Entidades.txt'
        EntFile = open(nameFile,'a')

        EntFile.write('---------- LISTA DE ENTIDADES -----------' +'\n')
        EntFile.write('* A continuación encontrara la lista de entidades presentes en las historias de usuario. Estan organizadas de la entidad mas frecuente a la menos frecuente *' +'\n')

        for entity in entities:
            EntFile.write(entity +'\n')
        
        EntFile.close()
        
        return nameFile

    def getClassFile(self, arrayClasses):
        nameFile = self.ResultsFolder + '/' + self.FileName + '-Clases.txt'

        classes = arrayClasses['Clases']
        relations = arrayClasses['Relaciones']

        ClassFile = open(nameFile,'a')
        for index in classes.keys():
            ClassFile.write('class ' + index + ' {' +'\n')

            if 'Atributos' in classes[index]:
                for attr in classes[index]['Atributos']:
                    ClassFile.write('   ' + attr +'\n')

            if 'Metodos' in classes[index]:
                for met in classes[index]['Metodos']:
                    ClassFile.write('   ' + met +'() \n')

            ClassFile.write('}' +'\n')
        
        for relation in relations:
            if relation[2] == 'ASOC':
                ClassFile.write(relation[0] + ' -- ' + relation[1] +'\n')

            if relation[2] == 'AGR':
                ClassFile.write(relation[0] + ' o-- ' + relation[1] +'\n')

            if relation[2] == 'COMP':
                ClassFile.write(relation[0] + ' *-- ' + relation[1] +'\n')

            if relation[2] == 'HER':
                ClassFile.write(relation[0] + ' <|-- ' + relation[1] +'\n')
            
            if relation[2] == 'DEP':
                ClassFile.write(relation[0] + ' ..> ' + relation[1] +'\n')
            

        ClassFile.close()

        system('python3 -m plantuml ' + nameFile)

        return nameFile

    def getCaseUseFile(self, CaseUseArray):
        nameFile = self.ResultsFolder + '/' + self.FileName + '-CasoUso'

        
        for actor in CaseUseArray.keys():
            actorFileName = nameFile + '-' + actor + '.txt'
            CaseFile = open(actorFileName,'a')
            CaseFile.write('left to right direction' +'\n')
            CaseFile.write('skinparam actorStyle awesome' +'\n')

            for case in CaseUseArray[actor]:
                CaseFile.write(':' + actor + ': --> (' + case + ')' +'\n')
            CaseFile.close()
            system('python3 -m plantuml ' + actorFileName)

        return nameFile
