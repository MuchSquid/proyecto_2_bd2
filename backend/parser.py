import re
from metodo import *
import json


class SQLParser:
    def __init__(self, query):
        self.query = query.strip()
        self.fields = []
        self.table = ""
        self.condition_field = ""
        self.condition_value = ""
        self.file_path = ""

    def parse_query(self):
        select_pattern = r"SELECT (.+) FROM (\w+) WHERE (.+) liketo '(.+)'"
        create_table_pattern = r"CREATE TABLE (\w+) FROM FILE \"(.+)\""

        create_match = re.match(create_table_pattern, self.query, re.IGNORECASE)
        if create_match:
            self.table = create_match.group(1).strip()
            self.file_path = create_match.group(2).strip()
            return "CREATE_TABLE"

        select_match = re.match(select_pattern, self.query, re.IGNORECASE)
        if select_match:
            self.fields = [field.strip() for field in select_match.group(1).split(",")]
            self.table = select_match.group(2).strip()
            self.condition_field = select_match.group(3).strip()
            self.condition_value = select_match.group(4).strip()
            return "SELECT"
        
        raise SyntaxError("Consulta SQL malformada")

    def get_parsed_query(self):
        return {
            "fields": self.fields,
            "table": self.table,
            "condition_field": self.condition_field,
            "condition_value": self.condition_value,
            "file_path": self.file_path
        }

def search(query, filename, top_k=5):
    query_terms = preprocess_text(query)
    
    with open(filename, 'r') as f:
        tf_idf_data = json.load(f)
        tf_idf_index = tf_idf_data['tfidf']
    
    query_vector = defaultdict(float)
    for term in query_terms:
        if term in tf_idf_index:
            query_vector[term] = 1 * math.log(len(tf_idf_index) / (1 + len(tf_idf_index[term])))
    
    query_norm = np.sqrt(sum(score ** 2 for score in query_vector.values()))

    norms = calcular_norma(tf_idf_data)
    
    similarities = {}
    for term, query_score in query_vector.items():
        if term in tf_idf_index:
            for doc_id, doc_score in tf_idf_index[term].items():
                similarities[doc_id] = similarities.get(doc_id, 0) + query_score * doc_score

    for doc_id in similarities:
        if doc_id in norms:
            similarities[doc_id] /= (query_norm * norms[doc_id]) if query_norm != 0 and norms[doc_id] != 0 else 0
    
    top_docs = sorted(similarities.items(), key=lambda item: item[1], reverse=True)[:top_k]
    return top_docs

def execute_query(parsed_query, filename, top_k):
    fields = parsed_query['fields']
    condition_value = parsed_query['condition_value']
    table_name = parsed_query['table']
    
    if table_name not in database:
        raise ValueError(f"Tabla '{table_name}' no encontrada.")

    top_docs = search(condition_value, filename, top_k)
    
    return top_docs

def explainAnalyze(query, filename, top_k):
    startTimer = time.time()
    parser = SQLParser(query)
    if parser.parse_query() == "SELECT":
        parsed_select = parser.get_parsed_query()
        results = execute_query(parsed_select, filename, top_k)
    else:
        print("Error al analizar la consulta.")
    
    end_timer = time.time()
    execution_time = end_timer - startTimer
    return results, execution_time

def save_index_to_json(tf_idf_index, filename="tfidf_index.json"):
    index_data = {
        "tfidf": tf_idf_index
    }
    with open(filename, 'w') as f:
        json.dump(index_data, f, indent=4)
    return filename

create_query = 'CREATE TABLE spotifyData FROM FILE "spotifyData.csv"'
parser = SQLParser(create_query)
if parser.parse_query() == "CREATE_TABLE":
    parsed_create = parser.get_parsed_query()
    create_table_from_file(parsed_create['table'], parsed_create['file_path'])

documents = [track['text'] for track in database['spotifyData']]

inverted_index, documents = build_inverted_index_and_tfidf(documents)

tf_idf_index = calculate_tf_idf(documents, inverted_index)

filename = save_index_to_json(tf_idf_index)

select_query = " SELECT track_name, track_artist FROM spotifyData WHERE lyrics liketo 'love'"
# select_query = "SELECT track_name, track_artist FROM spotifyData WHERE track_name liketo ' it love'"
parser = SQLParser(select_query)
top_k = 5

#explain analyze
results, execution_time = explainAnalyze(select_query, filename, top_k)

if results:
    print("Resultados de la consulta:")
    print(results)
    print(f"Tiempo de ejecución: {execution_time} segundos")

    # archivo_csv = 'dbprueba.csv'
    archivo_csv = 'spotifyData.csv'
    for doc_id, _ in results:
        pista = buscar_pista_en_csv(int(doc_id), archivo_csv)

        if pista:
            print(f"Detalles de la pista encontrada para el ID {doc_id}:")
            print(f"Track ID: {pista['track_id']}, Track Name: {pista['track_name']}")
        else:
            print(f"No se encontró ninguna pista con el ID {doc_id}.")


