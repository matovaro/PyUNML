txtArchivo = input('Ruta o nombre del archivo de texto: ')

with open(txtArchivo) as f_obj:
    lines = f_obj.readlines()

for line in lines:
    print(line)