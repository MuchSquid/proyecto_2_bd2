import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import StandardScaler


def loadData():
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
                "track_preview": fila["track_preview"],
                "lyrics": fila["lyrics"],
                "MFCC_Vector": punto,
                "duration": 30000,
            }
            puntos[track_id] = punto_info
        else:
            print(f"Advertencia: La fila {track_id} tiene un valor vacío o nulo en 'MFCC_Vector', se omite esta canción.")
    
    return puntos

def reducirPCA(df, n_components=15):
    mfcc_vectors = np.array([
        punto_info['MFCC_Vector'] 
        for punto_info in df.values()
    ])
 
    scaler = StandardScaler()
    mfcc_scaled = scaler.fit_transform(mfcc_vectors)
    
    pca = PCA(n_components=n_components)
    reduced_vectors = pca.fit_transform(mfcc_scaled)

    for i, (track_id, punto_info) in enumerate(df.items()):
        punto_info['Reduced_MFCC'] = reduced_vectors[i]
    
    return df, pca, scaler

def euclidean_distance(vector1, vector2):
    return np.linalg.norm(np.array(vector1) - np.array(vector2))

def knnSeq(query, C, k):
    distances = []
    
    for track_id, punto_info in C.items():
        vector = punto_info["Reduced_MFCC"]
        distance = euclidean_distance(query, vector)
        distances.append((distance, track_id))
    
    distances.sort(key=lambda x: x[0])
    return distances[:k]

def knnRange(query, C, radius):
    results = []
    
    for track_id, punto_info in C.items():
        vector = punto_info["Reduced_MFCC"]
        distance = euclidean_distance(query, vector)
        
        if distance <= radius:
            results.append((distance, track_id))
    
    return results


def main():
    puntos = loadData()
    puntos_reducidos, pca, scaler = reducirPCA(puntos)
    
    track_id_query = '4l0O19WwpjzbDGkdFtkQ6W'
    
    print("Búsqueda KNN:")
    if track_id_query in puntos_reducidos:
        query_vector = puntos_reducidos[track_id_query]["Reduced_MFCC"]
        k = 5 

        closest_songs = knnSeq(query_vector, puntos_reducidos, k)

        # print(f"Las {k} canciones más similares a la canción con track_id {track_id_query}:")
        for distance, track_id in closest_songs:
            song_info = puntos_reducidos[track_id]
            print(f"Distancia: {distance}, Track ID: {track_id}, Nombre: {song_info['track_name']}, Artista: {song_info['track_artist']}")
    else:
        print(f"La canción con track_id {track_id_query} no existe en la base de datos.")
        

    print("\nBúsqueda por rango:")

    if track_id_query in puntos_reducidos:
        query_vector = puntos_reducidos[track_id_query]["Reduced_MFCC"]
        radius = 6.0

        range_results = knnRange(query_vector, puntos_reducidos, radius)

        # print(f"Canciones dentro del rango de distancia {radius} a la canción con track_id {track_id_query}:")
        if range_results:
            for distance, track_id in range_results:
                song_info = puntos_reducidos[track_id]
                print(f"Distancia: {distance}, Track ID: {track_id}, Nombre: {song_info['track_name']}, Artista: {song_info['track_artist']}")
        else:
            print("No se encontraron canciones dentro del rango especificado.")
    else:
        print(f"La canción con track_id {track_id_query} no existe en la base de datos.")

if __name__ == "__main__":
    main()
