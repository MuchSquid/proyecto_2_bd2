import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from unidecode import unidecode
import numpy as np
import csv
import math
from collections import defaultdict
import string


# Diccionario para almacenar las "tablas" en memoria
database = {}

# Función para crear la tabla desde el archivo CSV
def create_table_from_file(table_name, file_path):
    with open(file_path, mode='r', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        records = []
        for row in csv_reader:
            # Concatenar las columnas de texto
            row['text'] = f"{row['track_name']} {row['track_artist']} {row['lyrics']}"
            records.append(row)
        database[table_name] = records
    print(f"Tabla '{table_name}' creada con datos de '{file_path}'")

# Preprocesamiento de texto
def preprocess_text(text):
    # Tokenizar y convertir a minúsculas
    palabras = nltk.word_tokenize(text.lower())
    
    # Cargar stopwords de varios idiomas
    stop_words = set()
    languages = [
        'english',    
        'spanish',    
        'french',     
        'german',     
        'italian',    
        'portuguese', 
        'russian',    
    ]
    for lang in languages:
        stop_words.update(stopwords.words(lang))

    # Filtrar palabras que sean alfabéticas, sin acentos, y que no sean stopwords
    palabras_filtradas = [unidecode(word) for word in palabras if word.isalpha() and unidecode(word) not in stop_words]
    
    # Lemmatizar las palabras filtradas
    lematizador = WordNetLemmatizer()
    palabras_lemmatizadas = [lematizador.lemmatize(word) for word in palabras_filtradas]
    
    return palabras_lemmatizadas

# Función para calcular el índice TF-IDF
def calculate_tf_idf(documents):
    N = len(documents)  # Total de documentos
    tf_idf_index = defaultdict(lambda: defaultdict(float))  # {term: {doc_id: tf-idf}}
    doc_term_freqs = defaultdict(lambda: defaultdict(int))  # {doc_id: {term: freq}}
    doc_lengths = defaultdict(float)  # {doc_id: norm}

    # Calcular frecuencia de término (TF)
    for doc_id, text in enumerate(documents):
        terms = preprocess_text(text)  # Usar el preprocesamiento aquí
        total_terms = len(terms)
        for term in terms:
            doc_term_freqs[doc_id][term] += 1

    # Calcular TF-IDF
    term_doc_count = defaultdict(int)  # {term: document count}
    for doc_id, term_freqs in doc_term_freqs.items():
        for term, freq in term_freqs.items():
            term_doc_count[term] += 1
            tf = freq / len(term_freqs)
            idf = math.log(N / (1 + term_doc_count[term]))  # Sumar uno para evitar log(0)
            tf_idf_index[term][doc_id] = tf * idf
    return tf_idf_index

# Construcción del índice invertido y TF-IDF
def build_inverted_index_and_tfidf():
    inverted_index = defaultdict(list)
    documents = []
    
    for track in database['spotifyData']:
        tokens = preprocess_text(track['text'])
        documents.append(track['text'])  # Almacenamos el texto original para TF-IDF
        for token in tokens:
            inverted_index[token].append(len(documents) - 1)  # Usamos el índice del documento
    
    return inverted_index, documents


def calcular_norma(tfidf_data):
    # Calcular la norma para cada documento
    norms = defaultdict(float)
    for term, docs in tfidf_data['tfidf'].items():
        for doc_id, score in docs.items():
            norms[doc_id] += score ** 2

    # Tomar la raíz cuadrada para obtener la norma
    for doc_id in norms:
        norms[doc_id] = np.sqrt(norms[doc_id])

    return norms  # Retornar el diccionario de normas

def buscar_pista_en_csv(index, archivo_csv):
    with open(archivo_csv, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for i, row in enumerate(reader):
            if i == index:  # Comparar con el índice recibido
                return {'track_id': row['track_id'], 'track_name': row['track_name']}
    return None  # Si no se encuentra