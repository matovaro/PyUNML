

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

    def ClassFile(self, arrayClasses):
        nameFile = self.FileName + '-Classes.txt'

        classes = arrayClasses['Clases']
        relations = arrayClasses['Relaciones']

        ClassFile = open(nameFile,'a')
        for index in classes.keys():
            ClassFile.write('class ' + index + ' {' +'\n')

            for attr in classes[index]['Atributos']:
                ClassFile.write('   ' + attr +'\n')

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

        return nameFile

    def CaseUseFile(self, CaseUseArray):
        nameFile = self.FileName + '-CaseUse.txt'

        CaseFile = open(nameFile,'a')

        CaseFile.write('skinparam actorStyle awesome' +'\n')
        for actor in CaseUseArray.keys():
            for case in CaseUseArray[actor]:
                CaseFile.write(':' + actor + ': --> (' + case + ')' +'\n')

        CaseFile.close()

        return nameFile