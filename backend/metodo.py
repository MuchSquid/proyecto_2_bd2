import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from unidecode import unidecode
import numpy as np
import csv
import math
from collections import defaultdict
import string
import json
import os
from datetime import datetime
import time

database = {}

def create_table_from_file(table_name, file_path):
    with open(file_path, mode='r', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        records = []
        for row in csv_reader:
            # Concatenar las columnas de texto
            row['text'] = f"{row['track_name']} {row['track_artist']} {row['lyrics']}"
            records.append(row)
        database[table_name] = records
    # Guardar el nombre de la tabla en un archivo
    with open("tablename.txt", mode='a', encoding='utf-8') as table_file:
        table_file.write(table_name + '\n')
        
    print(f"Tabla '{table_name}' creada con datos de '{file_path}'")

# Preprocesamiento de texto
def preprocess_text(text):
    palabras = nltk.word_tokenize(text.lower())
    
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

def calculate_tf_idf(documents, inverted_index):
    N = len(documents)
    tf_idf_index = defaultdict(lambda: defaultdict(float))

    for term, doc_ids in inverted_index.items():
        idf = math.log(N / (1 + len(doc_ids)))
        for doc_id in doc_ids:
            tf = doc_ids.count(doc_id) / len(doc_ids)
            tf_idf_index[term][doc_id] = tf * idf

    return tf_idf_index

def spimi_invert(documents, block_size_limit=1000):
    blocks = []
    current_block = defaultdict(list)
    block_counter = 0

    temp_dir = "BlocksTemporales"
    log_dir = "indexLogs"
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"spimi_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    
    def log_status(message):
        with open(log_file, 'a', encoding='utf-8') as f:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"[{timestamp}] {message}\n")
    
    log_status("Iniciando proceso SPIMI")
    log_status(f"Número total de documentos: {len(documents)}")

    for doc_id, text in enumerate(documents):
        terms = preprocess_text(text)
        
        for term in terms:
            current_block[term].append(doc_id)
            # Revisar si el tamaño del bloque supera el límite (por términos y postings)
            if sum(len(postings) for postings in current_block.values()) >= block_size_limit:
            #if len(current_block) >= block_size_limit:
                block_filename = f"block_{block_counter}.json"
                block_path = os.path.join(temp_dir, block_filename)
                
                with open(block_path, 'w', encoding='utf-8') as f:
                    json.dump(current_block, f)
                
                block_size = os.path.getsize(block_path) / 1024
                log_status(f"Bloque {block_counter} creado:")
                log_status(f"- Ubicación: {block_path}")
                log_status(f"- Tamaño: {block_size:.2f} KB")
                log_status(f"- Términos: {len(current_block)}")
                
                blocks.append(block_path)
                current_block = defaultdict(list)
                block_counter += 1

#si no esta vacio se crea bloque
    if current_block:
        block_filename = f"block_{block_counter}.json"
        block_path = os.path.join(temp_dir, block_filename)
        
        with open(block_path, 'w', encoding='utf-8') as f:
            json.dump(current_block, f)
        
        block_size = os.path.getsize(block_path) / 1024
        log_status(f"Bloque final {block_counter} creado:")
        log_status(f"- Ubicación: {block_path}")
        log_status(f"- Tamaño: {block_size:.2f} KB")
        log_status(f"- Términos: {len(current_block)}")
        
        blocks.append(block_path)

    log_status(f"SPIMI completado. Total de bloques: {len(blocks)}")
    return blocks

def merge_blocks(block_files):
    log_dir = "indexLogs"
    merge_log = os.path.join(log_dir, f"merge_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    
    def log_merge(message):
        with open(merge_log, 'a', encoding='utf-8') as f:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"[{timestamp}] {message}\n")
    
    log_merge(" Se incia merge")
    final_index = defaultdict(list)

    for block_file in block_files:
        if os.path.exists(block_file):
            with open(block_file, 'r', encoding='utf-8') as f:
                block_data = json.load(f)
                log_merge(f"Procesando bloque: {block_file}")
                log_merge(f" Términos en bloque: {len(block_data)}")
                
                for term, postings in block_data.items():
                    final_index[term].extend(postings)
                    final_index[term] = sorted(set(final_index[term]))  # Ordenar y eliminar duplicados
        else:
            log_merge(f"Bloque no encontrado: {block_file}")

    # Ordenar términos alfabéticamente
    sorted_final_index = {term: final_index[term] for term in sorted(final_index)}

    log_merge("Merge completo")
    log_merge(f"Términos totales en índice final: {len(final_index)}")
    
    return sorted_final_index

def clean_temp_blocks(wait_time=15):
    temp_dir = "BlocksTemporales"
    log_dir = "indexLogs"
    cleanup_log = os.path.join(log_dir, f"cleanup_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    
    def log_cleanup(message):
        with open(cleanup_log, 'a', encoding='utf-8') as f:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"[{timestamp}] {message}\n")
    
    log_cleanup(f"{wait_time} segundos pra eliminar BlocksTemporales...")
    # print(f"\nArchivos temporales en: {os.path.abspath(temp_dir)}")
    print(f"Se tiene {wait_time} segundos para verificar, antes de que se eliminen.")
    # print("Interrumpir programa, si se desea mantener.") //opcional?
    
    time.sleep(wait_time)
    
    for filename in os.listdir(temp_dir):
        file_path = os.path.join(temp_dir, filename)
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                log_cleanup(f"Archivo eliminado: {filename}")
        except Exception as e:
            log_cleanup(f"Error al eliminar {filename}: {str(e)}")
    
    try:
        os.rmdir(temp_dir)
        log_cleanup("Carpeta temporal eliminado")
    except Exception as e:
        log_cleanup(f"Error al eliminar directorio temporal: {str(e)}")

def build_inverted_index_and_tfidf(documents):
    print("\nIniciando construcción del índice invertido...")
    
    block_files = spimi_invert(documents)
    print(f"\nBloques creados: {len(block_files)}")
    print("Archivos temporales en 'BlocksTemporales'")
    print("Logs en 'indexLogs'")

    inverted_index = merge_blocks(block_files)

    tf_idf_index = calculate_tf_idf(documents, inverted_index)
    
    clean_temp_blocks()
    
    return inverted_index, tf_idf_index


def calcular_norma(tfidf_data):
    norms = defaultdict(float)
    for term, docs in tfidf_data['tfidf'].items():
        for doc_id, score in docs.items():
            norms[doc_id] += score ** 2

    for doc_id in norms:
        norms[doc_id] = np.sqrt(norms[doc_id])

    return norms  

def buscar_pista_en_csv(index, archivo_csv):
    with open(archivo_csv, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for i, row in enumerate(reader):
            if i == index:  
                return {'track_id': row['track_id'], 'track_name': row['track_name'], 'lyrics': row['lyrics']}
    return None
