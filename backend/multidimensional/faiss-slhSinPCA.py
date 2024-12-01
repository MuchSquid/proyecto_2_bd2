import numpy as np
import pandas as pd
import faiss
import time


def loadData(num_samples=None):
    file_path = 'c:/Users/Dario/Desktop/BD2/Proyecto3/spotifyCompleto.csv'
    df = pd.read_csv(file_path, on_bad_lines="skip")
    puntos = {}

    for i, fila in df.iterrows():
        track_id = fila["track_id"]
        mfcc_string = fila["MFCC_Vector"]

        if mfcc_string and not pd.isna(mfcc_string):
            try:
                mfcc_string = mfcc_string.replace("[", "").replace("]", "").replace("\n", "").strip()
                punto = mfcc_string.split()
                punto = [float(x) for x in punto if x]
            except ValueError:
                print(f"Error al convertir los vectores MFCC en la fila {track_id}, se omite esta canción.")
                continue 
            
            punto_info = {
                "track_id": track_id,
                "track_name": fila["track_name"],
                "track_artist": fila["track_artist"],
                "lyrics": fila["lyrics"],
                "mp3": fila["mp3"],
                "MFCC_Vector": punto,
                "duration": 30000,
            }
            puntos[track_id] = punto_info
        else:
            print(f"Advertencia: La fila {track_id} tiene un valor vacío o nulo en 'MFCC_Vector', se omite esta canción.")
    
    return puntos

def faiss_lsh(track_id, k, puntos, index):
    if track_id not in puntos:
        return None, None
    
    query_row = puntos[track_id]
    mfcc_query = np.array(query_row["MFCC_Vector"]).reshape(1, -1)
    
    distances, indices = index.search(mfcc_query, k + 1)
    similares = [(idx, dist) for dist, idx in zip(distances[0], indices[0]) if idx != list(puntos.keys()).index(track_id)]
    
    similares = similares[:k]
    
    if similares:
        filtered_indices, filtered_distances = zip(*similares)
        similar_tracks = [list(puntos.keys())[idx] for idx in filtered_indices]
        similar_info = [puntos[track_id] for track_id in similar_tracks]
        return similar_info, filtered_distances
    else:
        return [], []
    
    
def experimentoTiempo():
    dataSizes = [1000, 2000, 4000, 6000, 8000, 10000]
    k = 8

    for size in dataSizes:
        print(f"\nProcesando {size} datos...")
        puntos = loadData(num_samples=size)
        
        inicioTiempo = time.time()

        n_bits = 128
        dimension = 50
        index = faiss.IndexLSH(dimension, n_bits)

        mfcc_vectors = np.array([punto["MFCC_Vector"] for punto in puntos.values()]).astype('float32')

        index.train(mfcc_vectors)
        index.add(mfcc_vectors)

        track_id_query = '09nSCeCs6eYfAIJVfye1CE'

        # Búsqueda FAISS
        faiss_start_time = time.time()
        closest_songs, distances = faiss_lsh(track_id_query, k, puntos, index)
        faiss_end_time = time.time()

        print(f"Las {k} canciones más similares a la canción con track_id {track_id_query}:")
        for distance, track_id in zip(distances, closest_songs):
            song_info = track_id
            print(f"Distancia: {distance:.4f}, Track ID: {song_info['track_id']}, Nombre: {song_info['track_name']}, Artista: {song_info['track_artist']}")

        faisslshTime = (faiss_end_time - faiss_start_time) * 1000
        print(f"Tiempo de faiss con índice LSH: {faisslshTime:.2f} ms")

        totalTiempo = (faiss_end_time - inicioTiempo) * 1000
        print(f"Tiempo total de procesamiento (incluyendo FAISS): {totalTiempo:.2f} ms")

experimentoTiempo()
