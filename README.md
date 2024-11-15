# Proyecto 2 y 3 BD2

# Introducción

- Objetivo general: Desarrollar un sistema de base de datos multimedia integral que combine modelos relacionales y técnicas avanzadas de recuperación de información basadas en contenido, para optimizar la búsqueda y gestión eficiente de datos textuales y multimedia, con un enfoque particular en audio.

Dataset: https://drive.google.com/drive/folders/1mMIidLbl75S0r_4Nlg-9zSHkjrBGog5M?usp=drive_link 
Se utilizo un dataset que contiene mas de 15000 canciones con sus letras cada una y 14 atributos

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

# Frontend

Para el frontend se utilizo el framework de Astro con React para estilizar se uso nextui para las tablas,
Ademas se uso typescript para la conexión con fastapi del backend.

# Autores

|                     **Esteban Vasquez Grados**                   |                                 **Darío Ricardo Nuñes Villacorta**                                 |                       **Yamileth Yarel Rincón Tejada**                     |  **Flavia Ailen Mañuico Quequejana** |   **Maria Fernanda Surco Vergara**  |
|:----------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------:|:----:|
