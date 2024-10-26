export interface TrackData {
  track_id: string;
  track_name: string;
  track_artist: string;
  lyrics: string;
  track_popularity: number;
  track_album_id: string;
  track_album_name: string;
  track_album_release_date: string;
  playlist_name: string;
  playlist_id: string;
  playlist_genre: string;
  playlist_subgenre: string;
  danceability: number;
  energy: number;
  key: number;
  loudness: number;
  mode: number;
  speechiness: number;
  acousticness: number;
  instrumentalness: number;
  liveness: number;
  valence: number;  
  tempo: number;
  duration_ms: number;
  language: string;
}

export const defaultTrackData: TrackData = {
  track_id: '',
  track_name: '',
  track_artist: '',
  lyrics: '',
  track_popularity: 0,
  track_album_id: '',
  track_album_name: '',
  track_album_release_date: '',
  playlist_name: '',
  playlist_id: '',
  playlist_genre: '',
  playlist_subgenre: '',
  danceability: 0,
  energy: 0,
  key: 0,
  loudness: 0,
  mode: 0,
  speechiness: 0,
  acousticness: 0,
  instrumentalness: 0,
  liveness: 0,
  valence: 0,
  tempo: 0,
  duration_ms: 0,
  language: ''
};

export const sanitizeTrackData = (data: Partial<TrackData>): TrackData => {
  return {
    track_id: data.track_id ?? '',
    track_name: data.track_name ?? 'Not Found',
    track_artist: data.track_artist ?? '',
    lyrics: data.lyrics ?? '',
    track_popularity: data.track_popularity ?? 0,
    track_album_id: data.track_album_id ?? '',
    track_album_name: data.track_album_name ?? '',
    track_album_release_date: data.track_album_release_date ?? '',
    playlist_name: data.playlist_name ?? '',
    playlist_id: data.playlist_id ?? '',
    playlist_genre: data.playlist_genre ?? '',
    playlist_subgenre: data.playlist_subgenre ?? '',
    danceability: data.danceability ?? 0,
    energy: data.energy ?? 0,
    key: data.key ?? 0,
    loudness: data.loudness ?? 0,
    mode: data.mode ?? 0,
    speechiness: data.speechiness ?? 0,
    acousticness: data.acousticness ?? 0,
    instrumentalness: data.instrumentalness ?? 0,
    liveness: data.liveness ?? 0,
    valence: data.valence ?? 0,
    tempo: data.tempo ?? 0,
    duration_ms: data.duration_ms ?? 0,
    language: data.language ?? ''
  };
};
