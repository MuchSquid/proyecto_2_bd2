import psycopg2
import psycopg2.extras
import pandas as pd
import matplotlib.pyplot as plt
import random
import re


conn = psycopg2.connect(
    host="127.0.0.1",
    database="postgres",
    user="postgres",
    password="235689",
    port="5432"
)

def ejecutar_consulta(sql_str, select=False):
    try:
        cur = conn.cursor()
        cur.execute(sql_str)
        if select:
            rows = cur.fetchall()

            if 'EXPLAIN' in sql_str.upper():
                execution_plan = '\n'.join([row[0] for row in rows])
                return execution_plan
            else:
                df = pd.DataFrame(rows, columns=[desc[0] for desc in cur.description])
                return df
        else:
            conn.commit()   
            return None
    except Exception as e:
        print(f"Error executing query: {e}")
        print(f"SQL Query: {sql_str}")
        conn.rollback()
        raise e
    finally:
        cur.close()

sql_str = """
explain analyse
SELECT track_id, track_name, ts_rank(to_tsvector('spanish', lyrics), to_tsquery('spanish', 'Amor')) AS similitud
FROM spotifydata
WHERE to_tsvector('spanish', lyrics) @@ to_tsquery('spanish', 'Amor')
ORDER BY similitud DESC;
"""

print(ejecutar_consulta(sql_str, True))