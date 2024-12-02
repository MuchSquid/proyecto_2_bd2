from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from postgresql import ejecutar_consulta
import time
from parser import *
from multidimensional.knn import mainKnnPCA, experimentoTiempoKnnPCA
from multidimensional.knnSinPCA import mainKnn, experimentoTiempoKnn
from multidimensional.faiss_slh import experimentoTiempoFaissPCA
from multidimensional.faiss_slhSinPCA import experimentoTiempoFaiss
from multidimensional._rtree_algoritmo import mainRTreePCA
from multidimensional._rtree_algoritmo_sin_pca import mainRtree
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

@app.get("/get_time_local")
def get_time_local(
    q: str = Query(..., min_length=1, max_length=1000),
    k: int = Query(10, gt=0, lt=100000000000000),
): 
    select_query = f" SELECT track_name, track_artist, lyrics FROM spotifyData WHERE lyrics liketo '{q}'"
    parser = SQLParser(select_query)
    filename = "tfidf_index.json"
    archivo_csv = 'spotifyData.csv'
    
    results, execution_time = explainAnalyze(select_query, filename, k)
    
    if results:
        enhanced_results = []

        for doc_id, score in results:
            pista = buscar_pista_en_csv(int(doc_id), archivo_csv)
            
            if pista:
                pista_data = {
                    "track_id": pista['track_id'],
                    "track_name": pista['track_name'],
                    "lyrics": pista['lyrics'],
                    "similitud": score,
                }
                enhanced_results.append(pista_data)
            else:
                enhanced_results.append({
                    "track_id": doc_id,
                    "error": "No se encontró ninguna pista con el ID especificado."
                })
        
        
        return { "planning_time_ms": execution_time,
            "execution_time_ms": 0  ,
            "Query" : {select_query}}
    else:
        raise HTTPException(status_code=404, detail="No se encontraron resultados para la consulta.")

@app.get("/get_combined_data")
def ge_data(
    q: str = Query(..., min_length=1, max_length=1000),
    k: int = Query(10, gt=0, lt=100000000000000),
): 
    
    time_result = ejecutar_consulta(SqlTime(q, k), select=True)
    if "planning_time_ms" not in time_result or "execution_time_ms" not in time_result:
        raise HTTPException(status_code=500, detail="No se pudieron obtener los tiempos de planificación y ejecución.")
    
    
    data_result = ejecutar_consulta(SqlData(q, k), select=True)
    if isinstance(data_result, pd.DataFrame):
        data = data_result.to_dict(orient="records")
    elif isinstance(data_result, list):
        columns = ["track_id", "track_name", "lyrics", "similitud"]
        data = [dict(zip(columns, row)) for row in data_result]
    else:
        raise HTTPException(status_code=500, detail="No se pudo obtener la data en el formato esperado.")

    return {
        "planning_time_ms": time_result["planning_time_ms"],
        "execution_time_ms": time_result["execution_time_ms"],
        "query": SqlData(q, k),
        "data": data
    }


@app.get("/get_data_local")    
def get_data_local(
    q: str = Query(..., min_length=1, max_length=1000),
    k: int = Query(10, gt=0, lt=100000000000000),
): 
    select_query = f" SELECT track_name, track_artist, lyrics FROM spotifyData WHERE lyrics liketo '{q}'"
    parser = SQLParser(select_query)
    filename = "tfidf_index.json"
    archivo_csv = 'spotifyData.csv'
    
    results, execution_time = explainAnalyze(select_query, filename, k)
    
    if results:
        enhanced_results = []

        for doc_id, score in results:
            pista = buscar_pista_en_csv(int(doc_id), archivo_csv)
            
            if pista:
                pista_data = {
                    "track_id": pista['track_id'],
                    "track_name": pista['track_name'],
                    "lyrics": pista['lyrics'],
                    "similitud": score,
                }
                enhanced_results.append(pista_data)
            else:
                enhanced_results.append({
                    "track_id": doc_id,
                    "error": "No se encontró ninguna pista con el ID especificado."
                })
        
        
        return { "planning_time_ms": execution_time,
            "execution_time_ms": 0,
            "query": select_query,
            "data": enhanced_results
            }
    # , {       "planning_time_ms": execution_time,}
    else:
        raise HTTPException(status_code=404, detail="No se encontraron resultados para la consulta.")



@app.get("/get_knn_data_pca")
def ge_data(
    q: str = Query(..., min_length=1, max_length=1000),
    k: int = Query(10, gt=0, lt=100000000000000),
    tipo: str = Query("faiss", min_length=1, max_length=1000),
): 
    
    closest_songs, puntos_reducidos = mainKnnPCA(k, q, tipo)
    tiempo = experimentoTiempoKnnPCA(k, q, tipo)
    tiempototal = 0
    for tiem in tiempo:
        tiempototal += tiem
    
    enhanced_results = []
    for distance, track_id in closest_songs:
        song_info = puntos_reducidos[track_id]
        if song_info:
            pista_data = {
                "track_id": track_id,
                "track_name": song_info.get("track_name", "Unknown"),
                "track_artist": song_info.get("track_artist", "Unknown"),
                "similitud": distance,  # Aquí puedes cambiarlo si la similitud es otra métrica
            }
            enhanced_results.append(pista_data)
        else:
            enhanced_results.append({
                "track_id": track_id,
                "error": f"No se encontró información para track_id {track_id}."
            })

    return {
        "planning_time_ms": tiempototal,  
        "execution_time_ms": 0, 
        "query": q,
        "data": enhanced_results
    }
    
    
@app.get("/get_knn_data")
def ge_data(
    q: str = Query(..., min_length=1, max_length=1000),
    k: int = Query(10, gt=0, lt=100000000000000),
    tipo: str = Query("faiss", min_length=1, max_length=1000),
): 
    
    closest_songs, puntos_reducidos = mainKnn(k, q, tipo)
    tiempo = experimentoTiempoKnn(k, q, tipo)
    tiempototal = 0
    for tiem in tiempo:
        tiempototal += tiem
    
    enhanced_results = []
    for distance, track_id in closest_songs:
        song_info = puntos_reducidos[track_id]
        if song_info:
            pista_data = {
                "track_id": track_id,
                "track_name": song_info.get("track_name", "Unknown"),
                "track_artist": song_info.get("track_artist", "Unknown"),
                "similitud": distance, 
            }
            enhanced_results.append(pista_data)
        else:
            enhanced_results.append({
                "track_id": track_id,
                "error": f"No se encontró información para track_id {track_id}."
            })

    return {
        "planning_time_ms": tiempototal,  
        "execution_time_ms": 0, 
        "query": q,
        "data": enhanced_results
    }
    
@app.get("/get_faissPCA_data")
def ge_data(
    q: str = Query(..., min_length=1, max_length=1000),
    k: int = Query(10, gt=0, lt=100000000000000),
    tipo: str = Query("faiss", min_length=1, max_length=1000),
): 
    tiempototal = 0 
    closest_songs, distances, times = experimentoTiempoFaissPCA(k, q)
    for tiem in times:
        tiempototal += tiem
    
    
    enhanced_results = []
    for song_info, distance in zip(closest_songs, distances):
        enhanced_results.append({
            "track_id": song_info["track_id"],
            "track_name": song_info["track_name"],
            "track_artist": song_info["track_artist"],
            "similitud": float(distance), 
        })


    return {
        "planning_time_ms": int(tiempototal),
        "execution_time_ms": 0, 
        "query": q,
        "data": enhanced_results,
    }
    
    
@app.get("/get_faiss_data")
def ge_data(
    q: str = Query(..., min_length=1, max_length=1000),
    k: int = Query(10, gt=0, lt=100000000000000),
    tipo: str = Query("faiss", min_length=1, max_length=1000),
): 
    tiempototal = 0 
    closest_songs, distances, times = experimentoTiempoFaiss(k, q)
    for tiem in times:
        tiempototal += tiem
    
    
    enhanced_results = []
    for song_info, distance in zip(closest_songs, distances):
        enhanced_results.append({
            "track_id": song_info["track_id"],
            "track_name": song_info["track_name"],
            "track_artist": song_info["track_artist"],
            "similitud": float(distance), 
        })


    return {
        "planning_time_ms": int(tiempototal),
        "execution_time_ms": 0, 
        "query": q,
        "data": enhanced_results,
    }
    
    
@app.get("/get_rtree_data_pca")
def ge_data(
    q: str = Query(..., min_length=1, max_length=1000),
    k: int = Query(10, gt=0, lt=100000000000000),
    tipo: str = Query("faiss", min_length=1, max_length=1000),
): 
    
    closest_songs, puntos_reducidos, tiempo = mainRTreePCA(k, q, tipo)
    
    enhanced_results = []
    for distance, track_id in closest_songs:
        song_info = puntos_reducidos[track_id]
        if song_info:
            pista_data = {
                "track_id": track_id,
                "track_name": song_info.get("track_name", "Unknown"),
                "track_artist": song_info.get("track_artist", "Unknown"),
                "similitud": distance, 
            }
            enhanced_results.append(pista_data)
        else:
            enhanced_results.append({
                "track_id": track_id,
                "error": f"No se encontró información para track_id {track_id}."
            })

    return {
        "planning_time_ms": tiempo,  
        "execution_time_ms": 0, 
        "query": q,
        "data": enhanced_results
    }
    
    
@app.get("/get_rtree_data")
def ge_data(
    q: str = Query(..., min_length=1, max_length=1000),
    k: int = Query(10, gt=0, lt=100000000000000),
    tipo: str = Query("faiss", min_length=1, max_length=1000),
): 
    
    closest_songs, puntos_reducidos, tiempo = mainRtree(k, q, tipo)
    
    enhanced_results = []
    for distance, track_id in closest_songs:
        song_info = puntos_reducidos[track_id]
        if song_info:
            pista_data = {
                "track_id": track_id,
                "track_name": song_info.get("track_name", "Unknown"),
                "track_artist": song_info.get("track_artist", "Unknown"),
                "similitud": distance, 
            }
            enhanced_results.append(pista_data)
        else:
            enhanced_results.append({
                "track_id": track_id,
                "error": f"No se encontró información para track_id {track_id}."
            })

    return {
        "planning_time_ms": tiempo,  
        "execution_time_ms": 0, 
        "query": q,
        "data": enhanced_results
    }
    