import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import StandardScaler
from rtree import index
import time
import numpy as np

def loadData():
    file_path = '/Users/estebanmacbook/Document/Code/Astro/BD2_Proyecto/vectoresCaracteristicos/spotifyCompleto.csv'
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

    idx = index.Index(properties=prop, overwrite=True)
    track_to_id_map = {}

    for i, (track_id, punto_info) in enumerate(C.items()):
        vector = punto_info["Reduced_MFCC"]
        
        bounds = tuple(list(vector) * 2)
        
        idx.insert(i, bounds)
        track_to_id_map[i] = track_id 

    return idx, track_to_id_map


def knn_rtree(query, C, idx, track_to_id_map, k):
    try:
        search_range = tuple(list(query) * 2)
        
        results = []
        
        for item in idx.nearest(search_range, k):
            track_index = item if isinstance(item, int) else item.id
            track_id = track_to_id_map[track_index]     
            vector = C[track_id]["Reduced_MFCC"]
            distance = euclidean_distance(query, vector)
            results.append((distance, track_id))
    
        results.sort(key=lambda x: x[0])
        return results
    
    except Exception as e:
        print(f"Error en knn_rtree: {e}")
        return []

def range_search_rtree(rtree_index, query_vector, radius):
    min_bounds = [v - radius for v in query_vector]
    max_bounds = [v + radius for v in query_vector]
    
    rect = tuple(min_bounds) + tuple(max_bounds)
    results = list(rtree_index.intersection(rect)) 

    
    return results

def mainRTreePCA(k,track_id_query, tipo):
    puntos = loadData()
    puntos_reducidos, pca, scaler = reducirPCA(puntos)
    # print("Busqueda RTree:")
    # track_id_query = '0qYTZCo5Bwh1nsUFGZP3zn'
    
    if track_id_query in puntos_reducidos:
        query_vector = puntos_reducidos[track_id_query]["Reduced_MFCC"]

        print("\nConstruyendo R-tree con propiedades...")
        idx, track_to_id_map = construir_rtree_con_propiedades(puntos_reducidos, dimensions=len(query_vector))
        inicioTiempo = time.time()
        print("\nBúsqueda KNN con R-tree:")
        if(tipo == 'secuencial'):
            closest_songs_rtree = knn_rtree(query_vector, puntos_reducidos, idx, track_to_id_map, k)
            FinBusqueda = time.time()
            print(f"Tiempo de búsqueda KNN R-tree: {(FinBusqueda - inicioTiempo) * 1000:.2f} ms")
            tiempofinal = FinBusqueda - inicioTiempo
            for distance, track_id in closest_songs_rtree:
                song_info = puntos_reducidos[track_id]
                print(f"Distancia: {distance}, Track ID: {track_id}, Nombre: {song_info['track_name']}, Artista: {song_info['track_artist']}")
            
            return closest_songs_rtree, puntos_reducidos, tiempofinal


        
        print("Busqueda RTree por rango:")
        if(tipo == 'rango'):
            rangeInicio = time.time()
            range_results = range_search_rtree(idx, query_vector, k)
            rangeFin = time.time()
            print(f"Tiempo de búsqueda Range R-tree: {(rangeFin - rangeInicio) * 1000:.2f} ms")
            resultados = []
            for i in range_results:
                track_id = track_to_id_map[i]
                candidate_vector = puntos_reducidos[track_id]["Reduced_MFCC"]
                distancia = np.linalg.norm(np.array(candidate_vector) - np.array(query_vector))
                if distancia <= k:
                    resultados.append((distancia, track_id))

            resultados_ordenados = sorted(resultados, key=lambda x: x[0]) 
            print(f"Se encontraron {len(resultados_ordenados)} canciones dentro del rango:")
        
            for distancia, track_id in resultados_ordenados[:5]: 
                song_info = puntos[track_id]
                print(f"Distancia: {distancia:.4f}, Track ID: {track_id}, Nombre: {song_info['track_name']}, Artista: {song_info['track_artist']}")
            
            tiempofinal = rangeFin - rangeInicio
            return resultados_ordenados, puntos_reducidos, tiempofinal


    else:
        print(f"La canción con track_id {track_id_query} no existe en la base de datos.")





def experimentoTiempo():
    dataSizes = [1000, 2000, 4000, 6000, 8000, 10000]
    k = 8
    radius = 3.5
    
    for size in dataSizes:
        print(f"\nProcesando {size} datos...")
        
        puntos = loadData()
        puntos = dict(list(puntos.items())[:size])
        
        inicioTiempo = time.time()
        puntos_reducidos, pca, scaler = reducirPCA(puntos)
        pcaTiempo = time.time()
        print(f"Tiempo de reducción PCA: {(pcaTiempo - inicioTiempo) * 1000:.2f} ms")
        
        if not puntos_reducidos:
            print("No hay datos reducidos. Saltando esta iteración.")
            continue
        
        track_id_query = list(puntos_reducidos.keys())[0]
        query_vector = puntos_reducidos["09nSCeCs6eYfAIJVfye1CE"]["Reduced_MFCC"]
        
        rtreeConstruccionInicio = time.time()
        idx, track_to_id_map = construir_rtree_con_propiedades(puntos_reducidos, dimensions=len(query_vector))
        rtreeConstruccionTiempo = time.time()
        print(f"Tiempo de construcción R-tree: {(rtreeConstruccionTiempo - pcaTiempo) * 1000:.2f} ms")
        
        knnRtreeTiempo = time.time()
        closest_songs_rtree = knn_rtree(query_vector, puntos_reducidos, idx, track_to_id_map, k)
        knnRtreeFin = time.time()
        print(f"Tiempo de búsqueda KNN R-tree: {(knnRtreeFin - rtreeConstruccionTiempo) * 1000:.2f} ms")
        
        print(f"Las {k} canciones más similares a la canción con track_id {track_id_query}:")
        for distance, track_id in closest_songs_rtree:
            song_info = puntos_reducidos[track_id]
            print(f"Distancia: {distance:.4f}, Track ID: {track_id}, Nombre: {song_info['track_name']}, Artista: {song_info['track_artist']}")
        
        totalTiempo = knnRtreeFin - inicioTiempo
        print(f"Tiempo total: {totalTiempo * 1000:.2f} ms")


def experimentoTiempoRangeSearch():
    dataSizes = [1000, 2000, 4000, 6000, 8000, 10000]
    radius = 4
    track_id_query = '09nSCeCs6eYfAIJVfye1CE'

    for size in dataSizes:
        print(f"\nProcesando {size} datos...")
        puntos = loadData()
        puntos = dict(list(puntos.items())[:size]) 

        inicioTiempo = time.time()
        puntos_reducidos, pca, scaler = reducirPCA(puntos)
        pcaTiempo = time.time()
        print(f"Tiempo de reducción PCA: {(pcaTiempo - inicioTiempo) * 1000:.2f} ms")

        if track_id_query not in puntos:
            print(f"El track_id {track_id_query} no se encuentra en los datos. Saltando esta iteración.")
            continue

        query_vector = puntos[track_id_query]["Reduced_MFCC"]

        rtreeInicio = time.time()
        idx, track_to_id_map = construir_rtree_con_propiedades(puntos, dimensions=len(query_vector))
        rtreeFin = time.time()
        print(f"Tiempo de construcción R-tree: {(rtreeFin - pcaTiempo) * 1000:.2f} ms")

        rangeInicio = time.time()
        range_results = range_search_rtree(idx, query_vector, radius)
        rangeFin = time.time()
        print(f"Tiempo de búsqueda Range R-tree: {(rangeFin - rangeInicio) * 1000:.2f} ms")
        resultados = []
        for i in range_results:
            track_id = track_to_id_map[i]
            candidate_vector = puntos_reducidos[track_id]["Reduced_MFCC"]
            distancia = np.linalg.norm(np.array(candidate_vector) - np.array(query_vector))
            if distancia <= radius:
                resultados.append((distancia, track_id))

        resultados_ordenados = sorted(resultados, key=lambda x: x[0]) 
        print(f"Se encontraron {len(resultados_ordenados)} canciones dentro del rango:")
        
        for distancia, track_id in resultados_ordenados[:5]: 
            song_info = puntos[track_id]
            print(f"Distancia: {distancia:.4f}, Track ID: {track_id}, Nombre: {song_info['track_name']}, Artista: {song_info['track_artist']}")

        totalTiempo = rangeFin - inicioTiempo
        print(f"Tiempo total para {size} datos: {totalTiempo * 1000:.2f} ms")

# if __name__ == "__main__":
#     # experimentoTiempo()
#     main()
#     # experimentoTiempoRangeSearch()