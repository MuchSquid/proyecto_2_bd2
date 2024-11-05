import React, { useState, useEffect } from 'react';
import { MdDarkMode,MdLightMode } from "react-icons/md";

const ThemeToggleButton = () => {
  const [isDarkMode, setIsDarkMode] = useState(false);

  useEffect(() => {
    // Check the user's preference in localStorage or default to light mode
    const storedTheme = localStorage.getItem('theme');
    if (storedTheme === 'dark') {
      setIsDarkMode(true);
      document.documentElement.classList.add('dark');
    } else {
      setIsDarkMode(false);
      document.documentElement.classList.remove('dark');
    }
  }, []);

  const toggleDarkMode = () => {
    setIsDarkMode(!isDarkMode);
    if (!isDarkMode) {
      document.documentElement.classList.add('dark');
      localStorage.setItem('theme', 'dark');
    } else {
      document.documentElement.classList.remove('dark');
      localStorage.setItem('theme', 'light');
    }
  };

  return (
    <button
      onClick={toggleDarkMode}
      className={`p-2 rounded-full ${isDarkMode ? 'bg-gray-800 text-white' : 'bg-gray-200 text-black'}`}
    >
      {isDarkMode ? <MdLightMode /> : <MdDarkMode />}
    </button>
  );
};

export default ThemeToggleButton;
