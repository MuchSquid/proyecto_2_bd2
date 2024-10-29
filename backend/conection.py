from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from postgresql import ejecutar_consulta
import time


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def SqlData(keyValue, top_k=10):
    return f"""
explain analyse
SELECT track_id, track_name, ts_rank(to_tsvector('spanish', lyrics), to_tsquery('spanish', '${keyValue}')) AS similitud
FROM spotifydata
WHERE to_tsvector('spanish', lyrics) @@ to_tsquery('spanish', '${keyValue}')
ORDER BY similitud DESC
LIMIT {top_k};

"""
    
@app.get("/get_time")
def get_time(): 
    result = ejecutar_consulta(SqlData("    ", 10000), select=True)
    
    if "planning_time_ms" in result and "execution_time_ms" in result:
        return {
            "planning_time_ms": result["planning_time_ms"],
            "execution_time_ms": result["execution_time_ms"]
        }
    else:
        raise HTTPException(status_code=500, detail="No se pudieron obtener los tiempos de planificación y ejecución.")
