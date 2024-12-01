import numpy as np
import pandas as pd
import time

def loadData(num_samples=None):
    file_path = '../../vectoresCaracteristicos/spotifyCompleto.csv'
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

def euclidean_distance(vector1, vector2):
    return np.linalg.norm(np.array(vector1) - np.array(vector2))

def knnSeq(query, C, k):
    distances = []
    
    for track_id, punto_info in C.items():
        vector = punto_info["MFCC_Vector"]
        distance = euclidean_distance(query, vector)
        distances.append((distance, track_id))
    
    distances.sort(key=lambda x: x[0])
    return distances[:k]

def knnRange(query, C, radius):
    results = []
    
    for track_id, punto_info in C.items():
        vector = punto_info["MFCC_Vector"]
        distance = euclidean_distance(query, vector)
        
        if distance <= radius:
            results.append((distance, track_id))
    
    results.sort(key=lambda x: x[0])
    return results

def main():
    puntos = loadData()
    
    track_id_query = '09nSCeCs6eYfAIJVfye1CE'
    
    print("Búsqueda KNN:")
    if track_id_query in puntos:
        query_vector = puntos[track_id_query]["MFCC_Vector"]
        k = 10

        closest_songs = knnSeq(query_vector, puntos, k)

        for distance, track_id in closest_songs:
            song_info = puntos[track_id]
            print(f"Distancia: {distance}, Track ID: {track_id}, Nombre: {song_info['track_name']}, Artista: {song_info['track_artist']}")
    else:
        print(f"La canción con track_id {track_id_query} no existe en la base de datos.")
        

    print("\nBúsqueda por rango:")

    if track_id_query in puntos:
        query_vector = puntos[track_id_query]["MFCC_Vector"]
        radius = 3.5

        range_results = knnRange(query_vector, puntos, radius)

        if range_results:
            for distance, track_id in range_results:
                song_info = puntos[track_id]
                print(f"Distancia: {distance}, Track ID: {track_id}, Nombre: {song_info['track_name']}, Artista: {song_info['track_artist']}")
        else:
            print("No se encontraron canciones dentro del rango especificado.")
    
    else:
        print(f"La canción con track_id {track_id_query} no existe en la base de datos.")

def experimentoTiempo():
    dataSizes = [1000, 2000, 4000, 6000, 8000, 10000]
    k = 8
    radius = 10

    for size in dataSizes:
        print(f"\nProcesando {size} datos...")
        puntos = loadData(num_samples=size) 

        inicioTiempo = time.time()
        track_id_query = '09nSCeCs6eYfAIJVfye1CE'
        # print("Búsqueda KNN:")

        #knn secuencial
        query_vector = puntos[track_id_query]["MFCC_Vector"]
        closest_songs = knnSeq(query_vector, puntos, k)
        knnSecuencialTiempo = time.time()
        print(f"Tiempo de búsqueda KNN Secuencial: {(knnSecuencialTiempo - inicioTiempo) * 1000:.2f} ms")

            # print(f"Las {k} canciones más similares a la canción con track_id {track_id_query}:")
        # for distance, track_id in closest_songs:
        #     song_info = puntos_reducidos[track_id]
        #     print(f"Distancia: {distance}, Track ID: {track_id}, Nombre: {song_info['track_name']}, Artista: {song_info['track_artist']}")

        range_results = knnRange(query_vector, puntos, radius)
        knnRangeTiempo = time.time()
        print(f"Tiempo de búsqueda KNN Range: {(knnRangeTiempo - knnSecuencialTiempo) * 1000:.2f} ms")
        totalTiempo = knnRangeTiempo - inicioTiempo
        
        # for distance, track_id in range_results:
        #         song_info = puntos_reducidos[track_id]
        #         print(f"Distancia: {distance}, Track ID: {track_id}, Nombre: {song_info['track_name']}, Artista: {song_info['track_artist']}")
        
        print(f"Tiempo total: {totalTiempo * 1000:.2f} ms")

if __name__ == "__main__":
    # main()
    experimentoTiempo()