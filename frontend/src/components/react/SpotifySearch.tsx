import React, { useState, useEffect, useRef } from 'react';
import { Input, Button, Table, TableBody, TableCell, TableColumn, TableHeader, TableRow, Modal, ModalContent, ModalHeader, ModalBody } from '@nextui-org/react';
// @ts-ignore
import ColorThief from 'colorthief';
import AudioPlayer from './AudioPlayer';

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
  const [boxShadowColor, setBoxShadowColor] = useState<string>("rgba(0,0,0,0)");
  const imgRef = useRef<HTMLImageElement>(null);
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

  useEffect(() => {
    if (selectedTrack && imgRef.current) {
      const colorThief = new ColorThief();
      imgRef.current.onload = () => {
        const color = colorThief.getColor(imgRef.current!);
        setBoxShadowColor(`rgba(${color[0]}, ${color[1]}, ${color[2]}, 1)`);
      };
    }
  }, [selectedTrack]);

  return (
    <div className='flex flex-col gap-4'>
      <form onSubmit={handleSearch} className='flex gap-2'>
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
                      ref={imgRef}
                      crossOrigin="anonymous"
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
      <Modal 
        isOpen={isModalVisible} 
        onOpenChange={closeModal} 
        className='max-w-[450px] h-[720px] overflow-y-auto pb-5 bg-neutral-100 scrollbar-hide'
        hideCloseButton={true}
      >
        <ModalContent >
          {/* <ModalHeader className='text-3xl' >Song Details</ModalHeader> */}
          <div className='p-6 text-2xl font-bold'>
            Music<span className='text-red-600'>.ly</span>
          </div>
          
          <ModalBody className='flex justify-center pt-10'>
            {selectedTrack ? (
              <div className="flex justify-center items-center">
                <div>
                  <img 
                    src={selectedTrack.album.images[0]?.url || defaultImage} 
                    alt={selectedTrack.name || ''} 
                    className="max-w-xs rounded-sm"  
                    crossOrigin="anonymous"
                    ref={imgRef}
                    style={{ boxShadow: `0px 0px 50px ${boxShadowColor}` }} 
                  />
                  <p className="mt-10 mx-4 text-xl">{selectedTrack.name}</p>
                  <p className="text-md mx-4 ext-opacity-70">{selectedTrack.artists[0].name}</p>
                  {selectedTrack.preview_url && (
                  <AudioPlayer src={selectedTrack.preview_url} />
                  )}
                </div>
              </div>
            ) : (
              <p>Loading...</p>
            )}
            <div className='flex pt-10 px-5 justify-center items-center text-xl font-bolD'>
              Letra
            </div>
            <div className='flex px-5 justify-center text-center pb-6'>
            broken foreign when stretch number back ride doctor island provide wall dried younger cost particles anywhere snake smooth unhappy element upward buffalo then create
            wall dried younger cost particles anywhere snake smooth unhappy element upward buffalo then create
            </div>
          </ModalBody>
        </ModalContent>
      </Modal>
    </div>
  );
};

export default SpotifySearch2;