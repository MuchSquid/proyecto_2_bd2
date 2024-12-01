import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import StandardScaler
from rtree import index


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
                "lyrics": fila["lyrics"],
                "mp3": fila["mp3"],
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


def construir_rtree_con_propiedades(C, dimensions=15):
    prop = index.Property()
    prop.dimension = dimensions  
    prop.buffering_capacity = 10  
    prop.dat_extension = 'dat'
    prop.idx_extension = 'idx'

    idx = index.Index('puntos', properties=prop, overwrite=True)
    track_to_id_map = {}

    for i, (track_id, punto_info) in enumerate(C.items()):
        vector = punto_info["Reduced_MFCC"]
        idx.insert(i, vector + vector) 
        track_to_id_map[i] = track_id 

    return idx, track_to_id_map

def knn_rtree(query, C, idx, track_to_id_map, k):
    
    nearest = list(idx.nearest((*query, *query), k))
    results = []

    for i in nearest:
        track_id = track_to_id_map[i]
        vector = C[track_id]["Reduced_MFCC"]
        distance = euclidean_distance(query, vector)
        results.append((distance, track_id))

    results.sort(key=lambda x: x[0])  
    return results

def main():
    puntos = loadData()
    puntos_reducidos, pca, scaler = reducirPCA(puntos)
    
    track_id_query = '00Ia46AgCNfnXjzgH8PIKH'

    if track_id_query in puntos_reducidos:
        query_vector = puntos_reducidos[track_id_query]["Reduced_MFCC"]

        print("\nConstruyendo R-tree con propiedades...")
        idx, track_to_id_map = construir_rtree_con_propiedades(puntos_reducidos, dimensions=len(query_vector))

        print("\nBúsqueda KNN con R-tree:")
        k = 10
        closest_songs_rtree = knn_rtree(query_vector, puntos_reducidos, idx, track_to_id_map, k)

        for distance, track_id in closest_songs_rtree:
            song_info = puntos_reducidos[track_id]
            print(f"Distancia: {distance}, Track ID: {track_id}, Nombre: {song_info['track_name']}, Artista: {song_info['track_artist']}")
    else:
        print(f"La canción con track_id {track_id_query} no existe en la base de datos.")



if __name__ == "__main__":
    main()
