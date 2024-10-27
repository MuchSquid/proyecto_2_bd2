import React, { useState, useEffect, useRef } from 'react';
import { Input, Button, Table, TableBody, TableCell, TableColumn, TableHeader, TableRow, Modal, ModalContent, ModalHeader, ModalBody, ModalFooter } from '@nextui-org/react';
import { FaMagnifyingGlass} from "react-icons/fa6";

interface Track {
  id: string;
  name: string;
  album: { images: { url: string }[]; name: string };
  artists: { name: string }[];
  preview_url: string;
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
  const [selectedTrack, setSelectedTrack] = useState<Track | null>(null);
  const [isModalVisible, setIsModalVisible] = useState(false);
  const audioRef = useRef<HTMLAudioElement>(null);
  
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

  const handleRowClick = (track: Track) => {
    setSelectedTrack(track);
    setIsModalVisible(true);
  };


  const closeModal = () => {
    setIsModalVisible(false);
    setSelectedTrack(null);
  };
  

  return (
    <div className='flex flex-col gap-4'>
      <form onSubmit={handleSearch}  className='flex gap-2'>
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
        <div className='rounded-lg'>
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
                <TableRow key={track.id} onClick={() => handleRowClick(track)}>
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
       {/* <Modal isOpen={isModalVisible} onClose={closeModal}>
        {selectedTrack && (
          <div>
            <h3>{selectedTrack.name}</h3>
            <img
              src={selectedTrack.album.images[0]?.url || defaultImage}
              alt={selectedTrack.name}
              width="150"
              className='mb-4'
            />
            <p><strong>Album:</strong> {selectedTrack.album.name}</p>
            <p><strong>Artists:</strong> {selectedTrack.artists.map((artist) => artist.name).join(', ')}</p>
            <p><strong>Track ID:</strong> {selectedTrack.id}</p>
            <Button onClick={closeModal}>Close</Button>
          </div>
        )}
      </Modal> */}
      <Modal isOpen={isModalVisible} onOpenChange={closeModal} className='max-w-[1200px] h-[800px] overflow-y-auto pb-5'>
        <ModalContent>
          <ModalHeader>Song Details</ModalHeader>
          <ModalBody>
            {selectedTrack ? (
              <div className="flex">
                <div>
                  <img src={selectedTrack!.album.images[0]?.url || defaultImage} alt={selectedTrack!.name || ''} className="max-w-xs rounded-sm" />
                  <p className="mt-5">{selectedTrack!.name}</p>
                  <p className="text-sm text-opacity-70">{selectedTrack!.artists[0].name}</p>
                  {selectedTrack!.preview_url && (
                    <audio ref={audioRef} controls className="mt-5">
                      <source src={selectedTrack!.preview_url} type="audio/mpeg" />
                    </audio>
                  )}
                </div>
                <div className="ml-5">
                  <h3 className="text-green-500 font-semibold">Album:</h3>
                  <p>{selectedTrack!.album?.name}</p>
                  <Table aria-label="Album Tracks">
                    <TableHeader>
                      <TableColumn>#</TableColumn>
                      <TableColumn>Title</TableColumn>
                      <TableColumn>Artist(s)</TableColumn>
                    </TableHeader>
                    <TableBody>
                      {tracks.map((track, index) => (
                        <TableRow key={track.id}>
                          <TableCell>{index + 1}</TableCell>
                          <TableCell>{track.name}</TableCell>
                          <TableCell>{track.artists.map(artist => artist.name).join(', ')}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              </div>
            ) : (
              <p>Loading...</p>
            )}
            {/* <h3 className="text-green-500 font-bold mt-4">Lyrics</h3>
            <p>{selectedTrack!.selectedLyrics}</p> */}
          </ModalBody>
          {/* <ModalFooter>
            <Button color="danger" onClick={closeModal}>
              Close
            </Button>
          </ModalFooter> */}
        </ModalContent>
      </Modal>

    </div>
    
  );
};

export default SpotifySearch2;
