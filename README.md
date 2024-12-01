# Proyecto 2 y 3 BD2

# Introducción

- Objetivo general: Desarrollar un sistema de base de datos multimedia integral que combine modelos relacionales y técnicas avanzadas de recuperación de información basadas en contenido, para optimizar la búsqueda y gestión eficiente de datos textuales y multimedia, con un enfoque particular en audio.

Dataset: https://drive.google.com/drive/folders/1mMIidLbl75S0r_4Nlg-9zSHkjrBGog5M?usp=drive_link 
Se utilizo un dataset que contiene mas de 15000 canciones con sus letras cada una y 14 atributos

Dataset 2 de música - mp3: https://drive.google.com/drive/u/3/folders/1J8RrRfegDamA51AzghoOiHxKvjplVxTM


# Backend
### Índice Invertido

### Preprocesamiento

El preprocesamiento es una etapa clave en la creación de un índice invertido para garantizar que los datos textuales sean consistentes y relevantes para las tareas de búsqueda y recuperación. El proceso involucra las siguientes etapas principales:

1. **Tokenización**: 
   - El texto se divide en palabras (tokens) usando la función `nltk.word_tokenize`. Esto asegura que cada palabra pueda procesarse de manera individual.

2. **Normalización y limpieza**: 
   - Se convierten todos los textos a minúsculas para evitar problemas de sensibilidad al caso.
   - Se eliminan los caracteres no alfabéticos y los acentos utilizando `unidecode`, lo que unifica el formato de las palabras.

3. **Eliminación de *Stopwords***: 
   - Se eliminan palabras comunes y poco informativas (como "y", "el", "de") utilizando las listas de *stopwords* en varios idiomas (`english`, `spanish`, `french`, etc.) disponibles en NLTK. Esto reduce el ruido en el índice.

4. **Lematización**: 
   - Las palabras se reducen a su forma base (lematización) mediante `WordNetLemmatizer` para consolidar las variaciones de una misma raíz, como "correr" y "corriendo".

### **Función de Preprocesamiento**

```python
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
```

El resultado de esta etapa es una lista de términos limpios y normalizados listos para ser indexados.

### SPIMI
La función `spimi_invert` implementa el algoritmo **SPIMI** (*Single-Pass In-Memory Indexing*) para construir índices invertidos de manera eficiente, optimizando el uso de memoria. Este proceso implica dividir los documentos en bloques que se almacenan en memoria secundaria, asegurando que el límite de memoria disponible no se supere. Luego, se realiza un **merge** para combinar los bloques y producir un índice invertido final.

#### Flujo principal:

1. **Construcción del índice invertido por bloques:**  
   - Los términos de cada documento se añaden a un bloque en memoria.
   - Cuando el tamaño del bloque excede un límite predefinido, este se escribe en un archivo JSON en memoria secundaria.
   - Cada bloque incluye una lista de documentos (*posting list*) donde aparece cada término.

2. **Merge de bloques:**  
   - Los bloques creados se combinan en un único índice invertido final.
   - Durante este proceso, se actualizan métricas como la frecuencia de documentos (*df*) y las listas de documentos asociados a cada término.

3. **Cálculo del índice TF-IDF:**  
   - Se calcula el puntaje TF-IDF para cada término y documento utilizando la frecuencia del término (*tf*), frecuencia de documentos (*df*) y el total de documentos (*N*).  
   - Los puntajes TF-IDF se almacenan en un índice separado para facilitar búsquedas eficientes.

4. **Limpieza de bloques temporales:**  
   - Una vez completado el índice invertido y el cálculo de TF-IDF, se eliminan los archivos temporales para liberar espacio en disco.

#### Funciones principales:

- `spimi_invert(documents, block_size_limit)`: Procesa documentos y genera bloques temporales con índices invertidos parciales.
- `merge_blocks(block_files)`: Combina bloques en un único índice invertido final.
- `calculate_tf_idf(documents, inverted_index)`: Calcula los puntajes TF-IDF para el índice.
- `clean_temp_blocks()`: Elimina los bloques temporales creados durante el proceso.

El índice resultante se utiliza para realizar búsquedas eficientes mediante la similitud de coseno, devolviendo los documentos más relevantes para una consulta dada.

### Experimento (comparación de tiempos PostgreSQL vs Indice Invertido)

![img1](/img1.png)

# Índices Multidimensionales

## MFFC (Extracción de vectores característicos)


## KNN Sequential


## RTree



## FAISS



## Experimentación 
Usaremos el track_id `09nSCeCs6eYfAIJVfye1CE` para realziar la experimentación con los 3 índices multidimensionales y compararlos.

### Aplicando PCA

N | KNN Sequential | KNN Range| RTree | FAISS
---|---|---|---|---|
1000 | 44.94 ms | 39.18 ms | 1.00 ms | 19.96 ms
2000 | 48.14 ms | 46.14 ms | 2.00 ms | 19.00 ms
4000 | 51.34 ms | 46.18 ms | 3.00 ms  | 18.99 ms
6000 | 50.12 ms | 42.22 ms | 5.02 ms | 17.96 ms 
8000 | 53.66 ms | 47.65 ms | 6.52 ms | 18.00 ms
10000 | 46.71 ms | 41.34 ms | 7.00 ms | 21.00 ms

### Sin PCA
N | KNN Sequential | KNN Range| RTree | FAISS
---|---|---|---|---|
1000 | 86.31 ms | 87.92 ms | 2.02 ms | 60.96 ms
2000 | 99.70 ms | 95.12 ms | 3.13 ms  | 41.97 ms
4000 | 90.70 ms | 91.37 ms | 4.01 ms  | 38.99 ms 
6000 | 92.11 ms | 82.47 ms | 4.28 ms  | 40.00 ms
8000 | 87.61 ms | 77.95 ms | 5.00 ms | 39.97 ms 
10000 | 86.00 ms | 78.90 ms | 7.02 ms | 33.77 ms


# Frontend

- Para el frontend se utilizo el framework de Astro con React, ademas para estilizar el ui se utilizó la librería de NextUI la cual nos da tablas, botones, etc con un diseño responsivo y moderno.

- Para las conexiones con el backend se uso typescript para las conexiones con las apis de backend.

- Se uso tambien adicional las librerias de spline para las animaciones.

- Se uso la api de spotify para agrega imagenes y audios a las canciones seguin el id de cada canción que se obtiene de las canciones.

# Autores

|                     **Esteban Vasquez Grados**                   |                                 **Darío Ricardo Nuñes Villacorta**                                 |                       **Yamileth Yarel Rincón Tejada**                     |  **Flavia Ailen Mañuico Quequejana** |   **Maria Fernanda Surco Vergara**  |
|:----------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------:|:----:|
