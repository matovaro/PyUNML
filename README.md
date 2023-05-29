# PyUNML

Software implementado para la generación automática de diagramas de clase y diagramas de casos de uso a partir de historias de usuario, por medio de procesamiento de lenguaje natural y reconocimiento de patrones.

## Instalación y configuración

Para instalar las librerías y módulos requeridos para el funcionamiento del software, ejecute el archivo _installation.py_ con el siguiente comando:
```
python installation.py
```

Posteriormente, se debe configurar el sistema y las librerías importadas de acuerdo a lo requerido por el sistema, por lo cual se debe ejecutar el comando:
```
python setup.py
```

## Uso del software

Para usar el software, se debe ejecutar el archivo _generator.py_:
```
python generator.py
```

Tras lo cual, se solicitara el nombre o ruta del archivo de texto donde están contenidas las historias de usuario y este debe ser ingresado por el usuario:
```
Ruta o nombre del archivo de texto: USTest.txt
```

Una vez ingresado el archivo, el software iniciará el análisis automáticamente y procederá a generar los correspondientes diagramas de clase y casos de uso (Puede tardar 1 o 2 minutos).

## Referencia
- Tovar Onofre, M. (2023). Generación de diagramas de clase y casos de uso a partir de historias de usuario utilizando procesamiento de lenguaje natural. Universidad Nacional de Colombia. (https://repositorio.unal.edu.co/handle/unal/83672)
- Dataset: https://github.com/matovaro/PyUNML-DataSet
