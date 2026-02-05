// Virtuoso-inspired dark theme for Maestro
import { createTheme } from '@mui/material/styles';

const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#fbbf24', // Neon amber
      light: '#fcd34d',
      dark: '#f59e0b',
      contrastText: '#000000',
    },
    secondary: {
      main: '#06B6D4', // Neon cyan
      light: '#22d3ee',
      dark: '#0891b2',
      contrastText: '#000000',
    },
    error: {
      main: '#ef4444',
      light: '#f87171',
      dark: '#dc2626',
    },
    warning: {
      main: '#ff0066', // Neon red
      light: '#ff3385',
      dark: '#cc0052',
    },
    success: {
      main: '#10b981',
      light: '#34d399',
      dark: '#059669',
    },
    background: {
      default: '#000000',
      paper: '#111111',
    },
    text: {
      primary: '#e0e0e0',
      secondary: '#9ca3af',
    },
    divider: '#222222',
  },
  typography: {
    fontFamily: '"Inter", "Helvetica", "Arial", sans-serif',
    h1: {
      fontFamily: '"IBM Plex Mono", monospace',
      fontWeight: 700,
    },
    h2: {
      fontFamily: '"IBM Plex Mono", monospace',
      fontWeight: 700,
    },
    h3: {
      fontFamily: '"IBM Plex Mono", monospace',
      fontWeight: 600,
    },
    h4: {
      fontFamily: '"IBM Plex Mono", monospace',
      fontWeight: 600,
    },
    h5: {
      fontFamily: '"IBM Plex Mono", monospace',
      fontWeight: 500,
    },
    h6: {
      fontFamily: '"IBM Plex Mono", monospace',
      fontWeight: 500,
    },
    button: {
      fontFamily: '"IBM Plex Mono", monospace',
      fontWeight: 500,
      textTransform: 'none',
    },
    code: {
      fontFamily: '"IBM Plex Mono", monospace',
    },
  },
  components: {
    MuiCssBaseline: {
      styleOverrides: {
        body: {
          scrollbarWidth: 'thin',
          scrollbarColor: '#333333 #111111',
          '&::-webkit-scrollbar': {
            width: '8px',
          },
          '&::-webkit-scrollbar-track': {
            background: '#111111',
          },
          '&::-webkit-scrollbar-thumb': {
            backgroundColor: '#333333',
            borderRadius: '4px',
          },
        },
      },
    },
    MuiAppBar: {
      styleOverrides: {
        root: {
          backgroundColor: '#111111',
          borderBottom: '1px solid #222222',
          boxShadow: 'none',
        },
      },
    },
    MuiDrawer: {
      styleOverrides: {
        paper: {
          backgroundColor: '#111111',
          borderRight: '1px solid #222222',
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: '8px',
          padding: '10px 20px',
          transition: 'all 0.3s ease',
          '&:hover': {
            transform: 'translateY(-2px)',
          },
        },
        contained: {
          boxShadow: '0 0 20px rgba(251, 191, 36, 0.3)',
          '&:hover': {
            boxShadow: '0 0 30px rgba(251, 191, 36, 0.5)',
          },
        },
        outlined: {
          borderColor: '#fbbf24',
          color: '#fbbf24',
          '&:hover': {
            backgroundColor: 'rgba(251, 191, 36, 0.1)',
            borderColor: '#fbbf24',
          },
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundColor: '#111111',
          backgroundImage: 'none',
          border: '1px solid #222222',
          borderRadius: '12px',
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          backgroundColor: '#111111',
          border: '1px solid #222222',
          borderRadius: '12px',
          transition: 'all 0.3s ease',
          '&:hover': {
            borderColor: '#fbbf24',
            boxShadow: '0 0 20px rgba(251, 191, 36, 0.1)',
          },
        },
      },
    },
    MuiTextField: {
      styleOverrides: {
        root: {
          '& .MuiOutlinedInput-root': {
            backgroundColor: '#000000',
            borderRadius: '8px',
            '& fieldset': {
              borderColor: '#222222',
            },
            '&:hover fieldset': {
              borderColor: '#fbbf24',
            },
            '&.Mui-focused fieldset': {
              borderColor: '#fbbf24',
            },
          },
        },
      },
    },
    MuiSelect: {
      styleOverrides: {
        root: {
          backgroundColor: '#000000',
          borderRadius: '8px',
        },
      },
    },
    MuiTableCell: {
      styleOverrides: {
        root: {
          borderColor: '#222222',
        },
        head: {
          backgroundColor: '#111111',
          fontFamily: '"IBM Plex Mono", monospace',
          fontWeight: 600,
          color: '#fbbf24',
        },
      },
    },
    MuiTableRow: {
      styleOverrides: {
        root: {
          '&:hover': {
            backgroundColor: 'rgba(251, 191, 36, 0.05)',
          },
        },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: {
          fontFamily: '"IBM Plex Mono", monospace',
        },
      },
    },
    MuiListItemButton: {
      styleOverrides: {
        root: {
          borderRadius: '8px',
          margin: '4px 8px',
          '&:hover': {
            backgroundColor: 'rgba(251, 191, 36, 0.1)',
          },
          '&.Mui-selected': {
            backgroundColor: 'rgba(251, 191, 36, 0.15)',
            '&:hover': {
              backgroundColor: 'rgba(251, 191, 36, 0.2)',
            },
          },
        },
      },
    },
    MuiListItemIcon: {
      styleOverrides: {
        root: {
          color: '#9ca3af',
          minWidth: '40px',
        },
      },
    },
    MuiBadge: {
      styleOverrides: {
        colorSecondary: {
          backgroundColor: '#ff0066',
        },
      },
    },
    MuiDivider: {
      styleOverrides: {
        root: {
          borderColor: '#222222',
        },
      },
    },
    MuiIconButton: {
      styleOverrides: {
        root: {
          color: '#9ca3af',
          transition: 'all 0.2s ease',
          '&:hover': {
            color: '#fbbf24',
            backgroundColor: 'rgba(251, 191, 36, 0.1)',
          },
        },
      },
    },
    MuiLinearProgress: {
      styleOverrides: {
        root: {
          backgroundColor: '#222222',
          borderRadius: '4px',
        },
        bar: {
          backgroundColor: '#fbbf24',
        },
      },
    },
    MuiAlert: {
      styleOverrides: {
        root: {
          borderRadius: '8px',
        },
        standardSuccess: {
          backgroundColor: 'rgba(16, 185, 129, 0.1)',
          border: '1px solid #10b981',
        },
        standardError: {
          backgroundColor: 'rgba(239, 68, 68, 0.1)',
          border: '1px solid #ef4444',
        },
        standardWarning: {
          backgroundColor: 'rgba(251, 191, 36, 0.1)',
          border: '1px solid #fbbf24',
        },
        standardInfo: {
          backgroundColor: 'rgba(6, 182, 212, 0.1)',
          border: '1px solid #06B6D4',
        },
      },
    },
  },
});

export default theme;
