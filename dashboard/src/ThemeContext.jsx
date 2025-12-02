import React, { createContext, useState, useMemo, useContext, useEffect } from 'react';
import { createTheme, ThemeProvider } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';

const ColorModeContext = createContext({ toggleColorMode: () => {} });

export const useColorMode = () => useContext(ColorModeContext);

export const AppThemeProvider = ({ children }) => {
  // Check local storage for preference or default to light
  const [mode, setMode] = useState(() => {
    const savedMode = localStorage.getItem('themeMode');
    return savedMode ? savedMode : 'light';
  });

  const colorMode = useMemo(
    () => ({
      toggleColorMode: () => {
        setMode((prevMode) => {
          const newMode = prevMode === 'light' ? 'dark' : 'light';
          localStorage.setItem('themeMode', newMode);
          return newMode;
        });
      },
    }),
    [],
  );

  const theme = useMemo(
    () =>
      createTheme({
        palette: {
          mode,
          ...(mode === 'light'
            ? {
                // Light mode
                background: { default: '#f5f7fa', paper: '#ffffff' },
                text: { primary: '#1e293b', secondary: '#64748b' },
                primary: { main: '#3b82f6' },
              }
            : {
                // Dark mode
                background: { default: '#0f172a', paper: '#1e293b' },
                text: { primary: '#f1f5f9', secondary: '#94a3b8' },
                primary: { main: '#60a5fa' },
              }),
        },
        components: {
          MuiCard: {
            styleOverrides: {
              root: ({ theme }) => ({
                borderColor: theme.palette.mode === 'dark' ? '#334155' : '#e2e8f0',
              }),
            },
          },
        },
      }),
    [mode],
  );

  // Sync mode with body attribute for CSS usage
  useEffect(() => {
    document.body.setAttribute('data-theme', mode);
  }, [mode]);

  return (
    <ColorModeContext.Provider value={colorMode}>
      <ThemeProvider theme={theme}>
        <CssBaseline />
        {children}
      </ThemeProvider>
    </ColorModeContext.Provider>
  );
};
