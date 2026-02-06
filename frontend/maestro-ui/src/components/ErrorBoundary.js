// Error boundary component for catching and displaying React errors
import React from 'react';
import Box from '@mui/material/Box';
import Alert from '@mui/material/Alert';
import AlertTitle from '@mui/material/AlertTitle';
import Button from '@mui/material/Button';
import RefreshIcon from '@mui/icons-material/Refresh';

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null, errorInfo: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    this.setState({ errorInfo });
    // Log error to console for debugging
    console.error('ErrorBoundary caught an error:', error, errorInfo);
  }

  handleReload = () => {
    window.location.reload();
  };

  render() {
    if (this.state.hasError) {
      return (
        <Box
          sx={{
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center',
            minHeight: '200px',
            p: 3,
          }}
        >
          <Alert
            severity="error"
            sx={{
              maxWidth: 600,
              width: '100%',
              '& .MuiAlert-message': {
                width: '100%',
              },
            }}
            action={
              <Button
                color="inherit"
                size="small"
                startIcon={<RefreshIcon />}
                onClick={this.handleReload}
                sx={{
                  fontFamily: '"IBM Plex Mono", monospace',
                  '&:hover': {
                    backgroundColor: 'rgba(239, 68, 68, 0.2)',
                  },
                }}
              >
                Reload
              </Button>
            }
          >
            <AlertTitle sx={{ fontFamily: '"IBM Plex Mono", monospace', fontWeight: 600 }}>
              Something went wrong
            </AlertTitle>
            <Box
              component="pre"
              sx={{
                fontFamily: '"IBM Plex Mono", monospace',
                fontSize: '0.75rem',
                whiteSpace: 'pre-wrap',
                wordBreak: 'break-word',
                m: 0,
                mt: 1,
                color: 'text.secondary',
              }}
            >
              {this.state.error?.message || 'An unexpected error occurred'}
            </Box>
          </Alert>
        </Box>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
