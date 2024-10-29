import psycopg2
import psycopg2.extras
import pandas as pd
import re

conn = psycopg2.connect(
    host="127.0.0.1",
    database="postgres",
    user="postgres",
    password="235689",
    port="5432"
)

def ejecutar_consulta(sql_str, select=True):
    try:
        cur = conn.cursor()
        cur.execute(sql_str)
        if select:
            rows = cur.fetchall()

            if 'EXPLAIN' in sql_str.upper():
                execution_plan = '\n'.join([row[0] for row in rows])
                
                planning_time = re.search(r"Planning Time: (\d+\.\d+) ms", execution_plan)
                execution_time = re.search(r"Execution Time: (\d+\.\d+) ms", execution_plan)
                
                planning_time_ms = float(planning_time.group(1)) if planning_time else None
                execution_time_ms = float(execution_time.group(1)) if execution_time else None
                
                return {
                    "execution_plan": execution_plan,
                    "planning_time_ms": planning_time_ms,
                    "execution_time_ms": execution_time_ms
                }
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
