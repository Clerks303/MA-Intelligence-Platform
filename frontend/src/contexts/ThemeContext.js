import React, { createContext, useContext, useEffect, useState } from 'react';

const ThemeContext = createContext({
  theme: 'light',
  setTheme: () => {},
  toggleTheme: () => {}
});

export const useTheme = () => {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
};

export const ThemeProvider = ({ children }) => {
  const [theme, setTheme] = useState(() => {
    // Check for saved theme in localStorage or default to 'light'
    if (typeof window !== 'undefined') {
      const savedTheme = localStorage.getItem('theme');
      if (savedTheme) {
        return savedTheme;
      }
      // Check system preference
      return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    }
    return 'light';
  });

  useEffect(() => {
    const root = window.document.documentElement;
    
    // Remove previous theme class
    root.classList.remove('light', 'dark');
    
    // Add current theme class
    root.classList.add(theme);
    
    // Save to localStorage
    localStorage.setItem('theme', theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme(prevTheme => prevTheme === 'light' ? 'dark' : 'light');
  };

  const value = {
    theme,
    setTheme,
    toggleTheme
  };

  return (
    <ThemeContext.Provider value={value}>
      {children}
    </ThemeContext.Provider>
  );
};