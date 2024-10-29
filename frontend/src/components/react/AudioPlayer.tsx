import React, { useRef, useState, useEffect } from 'react';
import { FaPlay, FaPause, FaStepForward, FaStepBackward, FaRandom, FaRedo } from 'react-icons/fa';

interface AudioPlayerProps {
  src: string;
}

const AudioPlayer: React.FC<AudioPlayerProps> = ({ src }) => {
  const audioRef = useRef<HTMLAudioElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [progress, setProgress] = useState(0);
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);

  useEffect(() => {
    const audio = audioRef.current;

    if (audio) {
      audio.addEventListener('timeupdate', () => {
        setCurrentTime(audio.currentTime);
        setProgress((audio.currentTime / audio.duration) * 100);
      });

      audio.addEventListener('loadedmetadata', () => {
        setDuration(audio.duration);
      });

      return () => {
        audio.removeEventListener('timeupdate', () => {});
        audio.removeEventListener('loadedmetadata', () => {});
      };
    }
  }, []);

  const togglePlayPause = () => {
    const audio = audioRef.current;
    if (audio) {
      if (isPlaying) {
        audio.pause();
      } else {
        audio.play();
      }
      setIsPlaying(!isPlaying);
    }
  };

  const handleProgressChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const audio = audioRef.current;
    if (audio) {
      const newProgress = parseFloat(e.target.value);
      audio.currentTime = (audio.duration / 100) * newProgress;
      setProgress(newProgress);
    }
  };

  const formatTime = (time: number) => {
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60).toString().padStart(2, '0');
    return `${minutes}:${seconds}`;
  };

  return (
    <div className="flex flex-col items-center  text-black p-4 w-80 mt-2">
      <audio ref={audioRef} src={src} preload="metadata" />

      <div className="flex items-center justify-between w-full text-xs text-black">
        <span>{formatTime(currentTime)}</span>
        <input
          type="range"
          min="0"
          max="100"
          value={progress}
          onChange={handleProgressChange}
          className="w-3/4 h-1 bg-black rounded-lg appearance-none"
          style={{
            backgroundImage: `linear-gradient(to right, black ${progress}%, #827f7f ${progress}%)`,
          }}
        />
        <span>{formatTime(duration)}</span>
      </div>

      <div className="flex items-center gap-6 mt-4 p-3 rounded-lg shadow-md w-full justify-center">
        <button className="text-lg text-gray-600 hover:text-black transition-colors">
          <FaRedo />
        </button>

        <button className="text-2xl text-gray-600 hover:text-black transition-colors">
          <FaStepBackward />
        </button>

        <button
          onClick={togglePlayPause}
          className="text-3xl text-white hover:text-gray-200  bg-gray-800 p-4 rounded-full ransition-all"
        >
          {isPlaying ? <FaPause /> : <FaPlay />}
        </button>

        <button className="text-2xl text-gray-600 hover:text-black transition-colors">
          <FaStepForward />
        </button>

        <button className="text-lg text-gray-600 hover:text-black  transition-colors">
          <FaRandom />
        </button>
      </div>
    </div>
  );
};

export default AudioPlayer;
