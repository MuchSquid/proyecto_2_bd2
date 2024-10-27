import re
from metodo import *
import json


# Clase para representar el parser SQL
class SQLParser:
    def __init__(self, query):
        self.query = query.strip()
        self.fields = []
        self.table = ""
        self.condition_field = ""
        self.condition_value = ""
        self.file_path = ""

    # Función para analizar la consulta SQL
    def parse_query(self):
        # Regex para SELECT ... WHERE ... LIKE
        select_pattern = r"SELECT (.+) FROM (\w+) WHERE (.+) liketo '(.+)'"
        # Regex para CREATE TABLE ... FROM FILE ...
        create_table_pattern = r"CREATE TABLE (\w+) FROM FILE \"(.+)\""

        # Detectar y analizar `CREATE TABLE`
        create_match = re.match(create_table_pattern, self.query, re.IGNORECASE)
        if create_match:
            self.table = create_match.group(1).strip()
            self.file_path = create_match.group(2).strip()
            return "CREATE_TABLE"

        # Detectar y analizar `SELECT`
        select_match = re.match(select_pattern, self.query, re.IGNORECASE)
        if select_match:
            self.fields = [field.strip() for field in select_match.group(1).split(",")]
            self.table = select_match.group(2).strip()
            self.condition_field = select_match.group(3).strip()
            self.condition_value = select_match.group(4).strip()
            return "SELECT"
        
        raise SyntaxError("Consulta SQL malformada")

    # Retorna los datos de la consulta
    def get_parsed_query(self):
        return {
            "fields": self.fields,
            "table": self.table,
            "condition_field": self.condition_field,
            "condition_value": self.condition_value,
            "file_path": self.file_path
        }

def search(condition_value, filename, top_k):
    # Preprocesar la consulta para crear su vector TF-IDF
    query_terms = preprocess_text(condition_value)
    query_vector = defaultdict(float)

    # Cargar el archivo JSON con los datos de TF-IDF
    with open(filename, 'r') as f:
        tfidf_data = json.load(f)

    # Calcular el vector TF-IDF para la consulta
    for term in query_terms:
        if term in tfidf_data['tfidf']:
            for doc_id, score in tfidf_data['tfidf'][term].items():
                query_vector[doc_id] += score

    # Calcular la norma del vector de la consulta
    query_norm = np.sqrt(sum(score ** 2 for score in query_vector.values()))

    # Calcular similitudes de coseno y obtener los documentos relevantes
    norms = calcular_norma(tfidf_data)  # Calcular normas de documentos
    similarities = {}
    for doc_id, score in query_vector.items():
        if doc_id in norms:  # Asegurarse de que el documento tiene norma calculada
            similarity = score / (query_norm * norms[doc_id]) if query_norm != 0 and norms[doc_id] != 0 else 0
            similarities[doc_id] = similarity

    # Ordenar los documentos por similitud y obtener los top-k
    top_docs = sorted(similarities.items(), key=lambda item: item[1], reverse=True)[:top_k]
    return top_docs  # Retornar los documentos top y sus similitudes

# Función para ejecutar la búsqueda basada en la consulta
def execute_query(parsed_query, filename):
    # Obtenemos los campos y el valor de la condición
    fields = parsed_query['fields']
    condition_value = parsed_query['condition_value']
    table_name = parsed_query['table']

    top_k = 3
    
    # Verificamos si la tabla está en la base de datos
    if table_name not in database:
        raise ValueError(f"Tabla '{table_name}' no encontrada.")

    # Filtramos los resultados utilizando `search`
    top_docs = search(condition_value, filename, top_k)
    
    return top_docs

# Función para guardar el índice TF-IDF en un archivo JSON
def save_index_to_json(tf_idf_index, filename="tfidf_index.json"):
        #"norms": doc_lengths
    index_data = {
        "tfidf": tf_idf_index
    }
    with open(filename, 'w') as f:
        json.dump(index_data, f, indent=4)
    return filename

# Crear tabla desde archivo
create_query = 'CREATE TABLE spotifyData FROM FILE "dbprueba.csv"'
parser = SQLParser(create_query)
if parser.parse_query() == "CREATE_TABLE":
    parsed_create = parser.get_parsed_query()
    create_table_from_file(parsed_create['table'], parsed_create['file_path'])

# Construir el índice invertido y TF-IDF
inverted_index, documents = build_inverted_index_and_tfidf()

# Calcular el índice TF-IDF
tf_idf_index = calculate_tf_idf(documents)

# Guardar el índice en un archivo JSON
filename = save_index_to_json(tf_idf_index)

# Ejecutar consulta SELECT ... WHERE ... LIKE
select_query = "SELECT track_name, track_artist FROM spotifyData WHERE lyrics liketo 'trees Pangarap'"
parser = SQLParser(select_query)
if parser.parse_query() == "SELECT":
    parsed_select = parser.get_parsed_query()
    results = execute_query(parsed_select, filename)
    print("Resultados de la consulta:")
    print(results)

    # Buscar las pistas correspondientes en el CSV
    archivo_csv = 'dbprueba.csv'  # Asegúrate de poner el nombre correcto del archivo CSV
    for doc_id, _ in results:
        pista = buscar_pista_en_csv(int(doc_id), archivo_csv)  # Convertir a entero el doc_id
        if pista:
            print(f"Detalles de la pista encontrada para el ID {doc_id}:")
            print(f"Track ID: {pista['track_id']}, Track Name: {pista['track_name']}")
        else:
            print(f"No se encontró ninguna pista con el ID {doc_id}.")

    #for result in results:
    #    print(result)
