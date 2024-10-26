import re
from TF_IDF import *

# Definir una clase para representar las consultas
class SQLParser:
    def __init__(self, query):
        self.query = query.strip()
        self.fields = []
        self.table = ""
        self.condition = ""

    # Función principal para analizar la consulta
    def parse_query(self):
        # Regex para capturar el SELECT, FROM y WHERE con LIKE
        pattern = r"SELECT (.+) FROM (\w+) WHERE (.+) liketo '(.+)'"
        match = re.match(pattern, self.query, re.IGNORECASE)

        if match:
            # Extraemos los campos, la tabla, y la condición
            self.fields = [field.strip() for field in match.group(1).split(",")]
            self.table = match.group(2).strip()
            self.condition_field = match.group(3).strip()
            self.condition_value = match.group(4).strip()
        else:
            raise SyntaxError("Consulta malformada")

    # Función para devolver los resultados
    def get_parsed_query(self):
        return {
            "fields": self.fields,
            "table": self.table,
            "condition_field": self.condition_field,
            "condition_value": self.condition_value
        }

# Ejemplo de consulta
query = "SELECT love, cort FROM Movies WHERE query liketo 'locamente millonarios'"

# Creamos el objeto SQLParser
parser = SQLParser(query)

# Analizamos la consulta
parser.parse_query()

# Mostramos los resultados
parsed_query = parser.get_parsed_query()
print(parsed_query)


# Función para ejecutar la búsqueda basándose en la consulta
def execute_query(parsed_query, documents):
    # Extraer los campos que se desean recuperar
    fields = parsed_query['fields']
    condition_value = parsed_query['condition_value']

    # Realizamos la búsqueda utilizando el valor de la condición
    results = search(condition_value)

    # Aquí 'documents' es un diccionario con los contenidos de cada documento
    top_k_results = []
    for doc_id, score in results:
        doc = documents[doc_id]  # Obtenemos el documento
        doc_result = {field: doc[field] for field in fields if field in doc}
        doc_result['score'] = score  # Añadimos la similitud de coseno
        top_k_results.append(doc_result)

    return top_k_results

# Supongamos que estos son los documentos en nuestra "tabla"
documents = {
    1: {"title": "El amor en tiempos de guerra", "artist": "Autor 1", "content": "amor en tiempos de guerra..."},
    2: {"title": "Amor y paz", "artist": "Autor 2", "content": "una historia de paz..."},
    3: {"title": "La guerra del amor", "artist": "Autor 3", "content": "el conflicto del amor..."}
}

# Ejecutamos la consulta
query_results = execute_query(parsed_query, documents)
print(query_results)
