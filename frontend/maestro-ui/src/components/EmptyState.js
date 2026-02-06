// Empty state component for displaying when no data is available
import React from 'react';
import PropTypes from 'prop-types';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Button from '@mui/material/Button';
import ScienceIcon from '@mui/icons-material/Science';

function EmptyState({ icon: Icon = ScienceIcon, title, description, actionLabel, onAction }) {
  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        py: 6,
        px: 3,
        textAlign: 'center',
      }}
    >
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          width: 80,
          height: 80,
          borderRadius: '50%',
          backgroundColor: 'rgba(251, 191, 36, 0.1)',
          border: '1px solid rgba(251, 191, 36, 0.3)',
          mb: 3,
        }}
      >
        <Icon
          sx={{
            fontSize: 40,
            color: 'primary.main',
          }}
        />
      </Box>
      <Typography
        variant="h6"
        sx={{
          fontFamily: '"IBM Plex Mono", monospace',
          fontWeight: 600,
          color: 'text.primary',
          mb: 1,
        }}
      >
        {title}
      </Typography>
      {description && (
        <Typography
          variant="body2"
          sx={{
            color: 'text.secondary',
            maxWidth: 400,
            mb: onAction ? 3 : 0,
          }}
        >
          {description}
        </Typography>
      )}
      {onAction && actionLabel && (
        <Button
          variant="contained"
          color="primary"
          onClick={onAction}
          sx={{
            fontFamily: '"IBM Plex Mono", monospace',
          }}
        >
          {actionLabel}
        </Button>
      )}
    </Box>
  );
}

EmptyState.propTypes = {
  icon: PropTypes.elementType,
  title: PropTypes.string.isRequired,
  description: PropTypes.string,
  actionLabel: PropTypes.string,
  onAction: PropTypes.func,
};

export default EmptyState;
