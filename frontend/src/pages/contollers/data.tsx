import React, { useState, useEffect, useRef } from 'react';
import { Input, Button, Table, TableBody, TableCell, TableColumn, TableHeader, TableRow, Modal, ModalContent, ModalBody } from "@nextui-org/react";
// @ts-ignore
import ColorThief from 'colorthief';
import AudioPlayer from '../../components/react/AudioPlayer';
import { motion } from 'framer-motion';
import { SearchIcon } from '../../components/react/SearchIcon';


interface TiemposData {
  planning_time_ms: number;
  execution_time_ms: number;
  Query: string;
}

interface TableRowData {
  track_id: string;
  track_name: string;
  similitud: number;
  lyrics: string;
}

interface SpotifyTrack {
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

async function searchSpotifyTrack(accessToken: string, query: string): Promise<SpotifyTrack | null> {
  const response = await fetch(`https://api.spotify.com/v1/search?q=${encodeURIComponent(query)}&type=track&limit=1`, {
    headers: {
      Authorization: `Bearer ${accessToken}`,
    },
  });

  const data = await response.json();
  return data.tracks.items.length > 0 ? data.tracks.items[0] : null;
}

const Tiempos: React.FC = () => {
  const [data, setData] = useState<TiemposData | null>(null);
  const [tableData, setTableData] = useState<TableRowData[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [query, setQuery] = useState<string>('Amor');
  const [k, setK] = useState<number>(1000);
  const [isModalVisible, setIsModalVisible] = useState(false);
  const [selectedTrack, setSelectedTrack] = useState<SpotifyTrack | null>(null);
  const [lyrics, setLyrics] = useState<string | null>(null);
  const [accessToken, setAccessToken] = useState<string | null>(null);
  const [boxShadowColor, setBoxShadowColor] = useState<string>("rgba(0,0,0,0)");
  const [backgroundColor, setBackgroundColor] = useState<string>("rgba(0,0,0,0)");
  const [textColor, setTextColor] = useState<string>("black"); 
  const imgRef = useRef<HTMLImageElement>(null);

  useEffect(() => {
    async function fetchAccessToken() {
      const token = await getToken();
      setAccessToken(token);
    }
    fetchAccessToken();
  }, []);

  const fetchData = async () => {
  setLoading(true);
  setError(null); 
  try {
    const response = await fetch(`http://localhost:8000/get_time?q=${encodeURIComponent(query)}&k=${k}`);
    if (!response.ok) {
      throw new Error("Error en la respuesta de la API");
    }
    const result: TiemposData = await response.json();
    setData(result);

    const tableResponse = await fetch(`http://localhost:8000/get_data?q=${encodeURIComponent(query)}&k=${k}`);
    if (!tableResponse.ok) {
      throw new Error("Error en la respuesta de la API para la tabla");
    }
    const tableResult: TableRowData[] = await tableResponse.json();
    setTableData(tableResult);

  } catch (err) {
    const errorMessage = err instanceof Error ? err.message : "Error desconocido";
    setError(errorMessage);
  } finally {
    setLoading(false);
  }
};


const handleRowClick = async (track: TableRowData) => {
  if (!accessToken) return;

  const spotifyTrack = await searchSpotifyTrack(accessToken, track.track_name);
  setSelectedTrack(spotifyTrack);
  setLyrics(track.lyrics);

  if (spotifyTrack?.album.images[0]) {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.src = spotifyTrack.album.images[0].url;
    img.onload = () => {
      const colorThief = new ColorThief();
      const color = colorThief.getColor(img);
      const [r, g, b] = color;

      setBackgroundColor(`rgba(${r}, ${g}, ${b}, 1)`);
      setTextColor(isColorDark(r, g, b) ? "white" : "black");
    };
  }
};
    const isColorDark = (r: number, g: number, b: number) => {
      const brightness = (r * 299 + g * 587 + b * 114) / 1000;
      return brightness < 128;
    };
  const closeModal = () => {
    setIsModalVisible(false);
    setSelectedTrack(null);
    setLyrics(null);
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
    <div className="flex flex-row h-screen">
      <div className="flex-1 p-4 overflow-y-auto">
        <div className='pt-4'>
      </div>
        <form>
          <div className="flex flex-wrap md:flex-nowrap gap-4">
            <Input 
              label="Letra de una canción.." 
              onChange={(e) => setQuery(e.target.value)} 
              className="dark:border-black border-1 rounded-xl shadow-sm" 
              // endContent={
              //   <SearchIcon className="text-black/50 mb-2 dark:text-white/90 text-slate-400 pointer-events-none flex-shrink-0" />
              // }
            />
            <Input 
              type="number" 
              label="Límite" 
              onChange={(e) => setK(Number(e.target.value))} 
              className="dark:border-black border-1 rounded-xl shadow-sm" 
            />
            <Button type='submit' onClick={fetchData} disabled={loading} className="bg-neutral-100 border-1 text-black shadow-sm  dark:bg-black dark:border-gray-800 dark:text-white hover:shadow-md h-14">
            <SearchIcon></SearchIcon>
          </Button>
          </div>
          
          
        </form>
        <div className="mt-6">
          <Table aria-label="Resultados de la consulta" removeWrapper className=''>
            <TableHeader >
              <TableColumn className='bg-transparent'>Track ID</TableColumn>
              <TableColumn className='bg-transparent'>Nombre de la Canción</TableColumn>
              <TableColumn className='bg-transparent'>Similitud</TableColumn>
            </TableHeader>
            <TableBody>
              {tableData.map((row, index) => (
                <TableRow key={index} onClick={() => handleRowClick(row)}>
                  <TableCell>{row.track_id}</TableCell>
                  <TableCell>{row.track_name}</TableCell>
                  <TableCell>{row.similitud.toFixed(6)}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
        <h1 className="text-lg font-semibold my-4">Tiempos de Ejecución de la Consulta</h1>
          
          {error && <p style={{ color: 'red' }}>Error: {error}</p>}
          
          {data && (
            <div className="p-2 border-2 border-gray-300 mx-auto inline-block rounded-xl">
              <p><strong>Planning Time:</strong> {data.planning_time_ms} ms</p>
              <p><strong>Execution Time:</strong> {data.execution_time_ms} ms</p>
              <p><strong>Query:</strong> {data.Query}</p>
            </div>
          )}
      </div>

      
        <motion.div
          className="w-[30%] p-4 overflow-y-auto shadow-md"
          style={{ backgroundColor, color: textColor }}
          key={selectedTrack?.id} 
          initial={{ opacity: 0, x: 100 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: -100 }}
          transition={{ duration: 0.5 }}
        >
        {selectedTrack ? (
          <div className="flex flex-col items-center m-10">
            <img 
              src={selectedTrack.album.images[0]?.url || defaultImage} 
              alt={selectedTrack.name} 
              className="max-w-xs rounded-sm"  
              crossOrigin="anonymous"
              ref={imgRef}
              style={{ boxShadow: `0px 0px 50px ${boxShadowColor}` }} 
            />
            <p className="mt-10 text-xl font-semibold">{selectedTrack.name}</p>
            <p className="text-md opacity-70">{selectedTrack.artists.map(artist => artist.name).join(', ')}</p>
            {selectedTrack.preview_url && (
               <AudioPlayer src={selectedTrack.preview_url} textColor={textColor} backgroundColor={backgroundColor} />
            )}
            <div className='flex pt-10 px-5 justify-center items-center text-xl font-bold'>
              Letra
            </div>
            <div className='flex justify-center text-justify'>
              {lyrics}
            </div>
          </div>
        ) : (
          <div className="text-center text-gray-500 mt-40">
            <p>No song selected</p>
            <p>Select a song to view details</p>
          </div>
        )}
        </motion.div>
      
    </div>
  );
};

export default Tiempos;