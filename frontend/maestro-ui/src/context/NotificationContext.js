// Centralized toast notification system using MUI Snackbar via React Context
import React, { createContext, useContext, useState, useCallback } from 'react';
import Snackbar from '@mui/material/Snackbar';
import Alert from '@mui/material/Alert';

const NotificationContext = createContext(null);

export function NotificationProvider({ children }) {
  const [notification, setNotification] = useState({
    open: false,
    message: '',
    severity: 'info', // success | error | warning | info
  });

  const notify = useCallback((message, severity = 'info') => {
    setNotification({
      open: true,
      message,
      severity,
    });
  }, []);

  const handleClose = (event, reason) => {
    if (reason === 'clickaway') {
      return;
    }
    setNotification((prev) => ({ ...prev, open: false }));
  };

  return (
    <NotificationContext.Provider value={{ notify }}>
      {children}
      <Snackbar
        open={notification.open}
        autoHideDuration={6000}
        onClose={handleClose}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
      >
        <Alert
          onClose={handleClose}
          severity={notification.severity}
          variant="filled"
          sx={{ width: '100%' }}
        >
          {notification.message}
        </Alert>
      </Snackbar>
    </NotificationContext.Provider>
  );
}

export function useNotification() {
  const context = useContext(NotificationContext);
  if (!context) {
    throw new Error('useNotification must be used within a NotificationProvider');
  }
  return context;
}
