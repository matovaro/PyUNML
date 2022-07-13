from os import system

system('pip install -U spacy')
system('pip install stanza')

system('python3 -m spacy download es_core_news_lg')
system('pip install nltk')

system('export PATH=$PATH:/home/mike/.local/bin')

system('pip3 install -U scikit-learn')

system('pip install plantuml')
system('pip install git+https://github.com/SamuelMarks/python-plantuml#egg=plantuml')