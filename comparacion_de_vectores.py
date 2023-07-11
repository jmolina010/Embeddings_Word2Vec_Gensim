import gensim
from gensim.models import KeyedVectors
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity


# creacion del modelo para usar word2vec
# Carga del modelo Spanish-Billion-Corpus (300 dimensiones, formato binario)
def cargar_modelo():
    word2vec=KeyedVectors.load_word2vec_format('./SBW-vectors-300-min5.bin.gz', binary=True, encoding='utf-8')
    return word2vec


def obtener_vector(palabra, modelo):
    try:
        resultado = modelo[palabra]
    except KeyError:
        resultado = None
    return resultado

word2vec = cargar_modelo()

# ejemplo con elementos relacionados con agua y aves
palabras = ['agua', 'rio', 'embalse', 'mar', 'grifo', 'botella', 'pez', 'cocodrilo', 'gaviota', 'buitre']
tokens = [obtener_vector(token, word2vec) for token in palabras]
similitud = cosine_similarity(tokens)
rounded = np.round(similitud, decimals=3)
print(f'Comparacion de los embeddings de {palabras}.')
print(rounded)
print ('\n\n')

# ejemplo con cánidos
palabras = ['beagle', 'labrador', 'retriever', 'perro', 'dálmata', 'lobo', 'hiena']
tokens = [obtener_vector(token, word2vec) for token in palabras]
similitud = cosine_similarity(tokens)
rounded = np.round(similitud, decimals=3)
print(f'Comparacion de los embeddings de {palabras}.')
print(rounded)

