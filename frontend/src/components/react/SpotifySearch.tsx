import React, { useState, useEffect } from 'react';
import { Input, Button, Table, TableBody, TableCell, TableColumn, TableHeader, TableRow } from '@nextui-org/react';

interface Track {
  id: string;
  name: string;
  album: { images: { url: string }[]; name: string };
  artists: { name: string }[];
}

const defaultImage = "https://via.placeholder.com/150"; 

async function getToken() {
  const clientId = "cdce14d45a9642a6a218e361e63a1c92";
  const clientSecret = "9b86e1effcc641e492fbc44a2fc81b15";
  const response = await fetch("https://accounts.spotify.com/api/token", {
    method: "POST",
    headers: {
      "Content-Type": "application/x-www-form-urlencoded",
    },
    body: new URLSearchParams({
      grant_type: "client_credentials",
      client_id: clientId,
      client_secret: clientSecret,
    }),
  });
  const data = await response.json();
  return data.access_token;
}
 
async function searchSpotifyTracks(accessToken: string, query: string): Promise<Track[]> {
  const response = await fetch(`https://api.spotify.com/v1/search?q=${encodeURIComponent(query)}&type=track`, {
    headers: {
      Authorization: `Bearer ${accessToken}`,
    },
  });

  const data = await response.json();
  return data.tracks.items;
}

const SpotifySearch2: React.FC = () => {
  const [query, setQuery] = useState('');
  const [tracks, setTracks] = useState<Track[]>([]);
  const [accessToken, setAccessToken] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    async function fetchAccessToken() {
      const token = await getToken();
      setAccessToken(token);
    }

    fetchAccessToken();
  }, []);

  const handleSearch = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!accessToken) return;

        
    setLoading(true);
    const tracks = await searchSpotifyTracks(accessToken, query);
    setTracks(tracks);
    setLoading(false);
  };

  return (
    <div>
      <form onSubmit={handleSearch}  className='flex justify-center gap-2'>
        <Input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search your favourite song"
          className='max-w-sm border-1 rounded-xl shadow-sm'
        />
        <Button type="submit" className='bg-neutral-100 border-1 text-black shadow-sm'>Search</Button>
      </form>
      {loading ? (
        <p>Loading...</p>
      ) : (
        <div className='p-5 rounded-lg'>
          <Table aria-label="Track Results">
            <TableHeader>
              <TableColumn>Image</TableColumn>
              <TableColumn>Song</TableColumn>
              <TableColumn>Album Name</TableColumn>
              <TableColumn>Artist(s)</TableColumn>
              <TableColumn>Id</TableColumn>
            </TableHeader>
            <TableBody>
              {tracks.map((track) => (
                <TableRow key={track.id}>
                  <TableCell>
                    <img 
                      src={track.album.images[0]?.url || defaultImage} 
                      alt={track.name} 
                      width="50" 
                    />
                  </TableCell>
                  <TableCell>{track.name}</TableCell>
                  <TableCell>{track.album.name}</TableCell>
                  <TableCell>{track.artists.map((artist) => artist.name).join(', ')}</TableCell>
                  <TableCell>{track.id}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      )}
    </div>
  );
};

export default SpotifySearch2;
