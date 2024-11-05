from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from postgresql import ejecutar_consulta
import time
import pandas as pd

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def SqlTime(keyValue, top_k=10):
    keyValue = keyValue.replace(" ", " & ")
    return f"""
    explain analyse
    SELECT track_id, track_name, lyrics, ts_rank(to_tsvector('spanish', lyrics), to_tsquery('spanish', '{keyValue}')) AS similitud
    FROM spotifydata
    WHERE to_tsvector('spanish', lyrics) @@ to_tsquery('spanish', '{keyValue}')
    ORDER BY similitud DESC
    LIMIT {top_k};
    """

def SqlData(keyValue, top_k=10):
    keyValue = keyValue.replace(" ", " & ")
    return f"""
    SELECT track_id, track_name, lyrics, ts_rank(to_tsvector('spanish', lyrics), to_tsquery('spanish', '{keyValue}')) AS similitud
    FROM spotifydata
    WHERE to_tsvector('spanish', lyrics) @@ to_tsquery('spanish', '{keyValue}')
    ORDER BY similitud DESC
    LIMIT {top_k};
    """

    
@app.get("/get_time")
def get_time(
    q: str = Query(..., min_length=1, max_length=1000),
    k: int = Query(10, gt=0, lt=100000000000000),
): 
    
    result = ejecutar_consulta(SqlTime(q, k), select=True)
    
    if "planning_time_ms" in result and "execution_time_ms" in result:
        return {
            "planning_time_ms": result["planning_time_ms"],
            "execution_time_ms": result["execution_time_ms"],
            "Query" : SqlData(q, k)
        }
    else:
        raise HTTPException(status_code=500, detail="No se pudieron obtener los tiempos de planificación y ejecución.")


@app.get("/get_data")
def get_data(
    q: str = Query(..., min_length=1, max_length=1000),
    k: int = Query(10, gt=0, lt=100000000000000),
): 
    # Ejecutar la consulta
    result = ejecutar_consulta(SqlData(q, k), select=True)
    
    # Verificar si el resultado es un DataFrame y convertirlo a JSON
    if isinstance(result, pd.DataFrame):
        return result.to_dict(orient="records")
    
    # Si el resultado es una lista de tuplas, convertir a lista de diccionarios
    elif isinstance(result, list):
        columns = ["track_id", "track_name", "similitud", "lyrics"]
        return [dict(zip(columns, row)) for row in result]
    
    else:
        raise HTTPException(status_code=500, detail="No se pudo obtener la data en el formato esperado.")