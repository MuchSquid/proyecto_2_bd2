import numpy as np
import pandas as pd
import faiss

df = pd.read_csv('spotifyCompleto2.csv')

def parse_mfcc_vector(mfcc_str):
    return np.fromstring(mfcc_str.strip('[]'), sep=' ') if mfcc_str else np.array([])

df['MFCC_Vector'] = df['MFCC_Vector'].apply(parse_mfcc_vector)
df = df[df['MFCC_Vector'].apply(lambda x: x.size > 0)]

mfcc_vectors = np.vstack(df['MFCC_Vector'].values).astype('float32')
index = faiss.IndexFlatL2(mfcc_vectors.shape[1])
index.add(mfcc_vectors)

def buscar_similares_por_track_id(track_id, k=5):
    if track_id not in df['track_id'].values:
        return None, None
    query_row = df[df['track_id'] == track_id].iloc[0]

    mfcc_query = query_row['MFCC_Vector'].reshape(1, -1)
    query_index = df.index.get_loc(query_row.name)

    distances, indices = index.search(mfcc_query, k + 1)
    similares = [(idx, dist) for dist, idx in zip(distances[0], indices[0]) if idx != query_index]
    filtered_indices, filtered_distances = zip(*similares[:k]) if similares else ([], [])
    
    return df.iloc[list(filtered_indices)], filtered_distances

consulta_track_id = "1Y373MqadDRtclJNdnUXVc"
canciones_similares, distancias = buscar_similares_por_track_id(consulta_track_id)

consulta_info = df[df['track_id'] == consulta_track_id][['track_name', 'track_artist']]
print("Canción de consulta:")
print(consulta_info)

print("\nCanciones similares:")
if canciones_similares is not None and not canciones_similares.empty:
    print(canciones_similares[['track_name', 'track_artist', 'track_id']])
    print("\nDistancias:")
    print(distancias)
