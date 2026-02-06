// Optuna Dashboard - Embeds Optuna dashboard via iframe or shows fallback
import React from 'react';
import Box from '@mui/material/Box';
import Alert from '@mui/material/Alert';
import AlertTitle from '@mui/material/AlertTitle';
import Link from '@mui/material/Link';
import SettingsIcon from '@mui/icons-material/Settings';

const OPTUNA_DASHBOARD_URL = process.env.REACT_APP_OPTUNA_DASHBOARD_URL;

export default function OptunaDashboard({ studyName, height = 600 }) {
  // Show info alert if Optuna dashboard URL is not configured
  if (!OPTUNA_DASHBOARD_URL) {
    return (
      <Alert
        severity="info"
        icon={<SettingsIcon />}
        sx={{
          '& .MuiAlert-message': {
            width: '100%',
          },
        }}
      >
        <AlertTitle>Optuna Dashboard Not Configured</AlertTitle>
        <Box sx={{ mt: 1 }}>
          To enable the embedded Optuna dashboard, set the{' '}
          <Box
            component="code"
            sx={{
              backgroundColor: 'rgba(251, 191, 36, 0.1)',
              px: 1,
              py: 0.5,
              borderRadius: 1,
              fontFamily: '"IBM Plex Mono", monospace',
              fontSize: '0.85em',
            }}
          >
            REACT_APP_OPTUNA_DASHBOARD_URL
          </Box>{' '}
          environment variable in your <code>.env</code> file.
        </Box>
        <Box sx={{ mt: 2, fontSize: '0.9em', color: 'text.secondary' }}>
          Example:{' '}
          <Box
            component="code"
            sx={{
              fontFamily: '"IBM Plex Mono", monospace',
              fontSize: '0.9em',
            }}
          >
            REACT_APP_OPTUNA_DASHBOARD_URL=http://localhost:8080
          </Box>
        </Box>
        <Box sx={{ mt: 2 }}>
          <Link
            href="https://optuna-dashboard.readthedocs.io/"
            target="_blank"
            rel="noopener noreferrer"
            sx={{ color: 'primary.main' }}
          >
            Learn more about Optuna Dashboard
          </Link>
        </Box>
      </Alert>
    );
  }

  // Build the dashboard URL with optional study filter
  const dashboardUrl = studyName
    ? `${OPTUNA_DASHBOARD_URL}?study=${encodeURIComponent(studyName)}`
    : OPTUNA_DASHBOARD_URL;

  return (
    <Box
      sx={{
        width: '100%',
        height: height,
        border: '1px solid',
        borderColor: 'divider',
        borderRadius: 2,
        overflow: 'hidden',
        backgroundColor: 'background.paper',
      }}
    >
      <iframe
        src={dashboardUrl}
        title="Optuna Dashboard"
        width="100%"
        height="100%"
        style={{
          border: 'none',
          backgroundColor: '#111111',
        }}
        sandbox="allow-scripts allow-same-origin allow-forms allow-popups"
      />
    </Box>
  );
}
