/**
 * OptimizationProgress - Rich progress visualization component
 *
 * Displays optimization progress with:
 * - Status chip (pending/running/completed/failed)
 * - Connection indicator (Live/Polling)
 * - Styled progress bar with percentage
 * - Expandable fold visualization
 * - Stop button for cancellation
 */

import React, { useState } from 'react';
import Box from '@mui/material/Box';
import Chip from '@mui/material/Chip';
import Collapse from '@mui/material/Collapse';
import IconButton from '@mui/material/IconButton';
import LinearProgress from '@mui/material/LinearProgress';
import Typography from '@mui/material/Typography';
import Tooltip from '@mui/material/Tooltip';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ExpandLessIcon from '@mui/icons-material/ExpandLess';
import StopIcon from '@mui/icons-material/Stop';
import WifiIcon from '@mui/icons-material/Wifi';
import SyncIcon from '@mui/icons-material/Sync';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ErrorIcon from '@mui/icons-material/Error';
import HourglassEmptyIcon from '@mui/icons-material/HourglassEmpty';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import { useNotification } from '../context/NotificationContext';

/**
 * Status configuration with colors and icons
 */
const STATUS_CONFIG = {
  pending: {
    color: 'default',
    bgColor: 'rgba(156, 163, 175, 0.1)',
    borderColor: '#9ca3af',
    icon: HourglassEmptyIcon,
    label: 'Pending',
  },
  running: {
    color: 'primary',
    bgColor: 'rgba(251, 191, 36, 0.1)',
    borderColor: '#fbbf24',
    icon: PlayArrowIcon,
    label: 'Running',
  },
  completed: {
    color: 'success',
    bgColor: 'rgba(16, 185, 129, 0.1)',
    borderColor: '#10b981',
    icon: CheckCircleIcon,
    label: 'Completed',
  },
  failed: {
    color: 'error',
    bgColor: 'rgba(239, 68, 68, 0.1)',
    borderColor: '#ef4444',
    icon: ErrorIcon,
    label: 'Failed',
  },
};

/**
 * Single fold box component
 */
function FoldBox({ index, status, isCurrent }) {
  const colors = {
    completed: '#10b981', // success.main
    current: '#fbbf24',   // primary.main
    pending: '#333333',   // grey.800
  };

  const getColor = () => {
    if (status === 'completed') return colors.completed;
    if (isCurrent) return colors.current;
    return colors.pending;
  };

  const color = getColor();
  const isActive = status === 'completed' || isCurrent;

  return (
    <Tooltip title={`Fold ${index + 1}: ${isCurrent ? 'In Progress' : status}`}>
      <Box
        sx={{
          width: 24,
          height: 24,
          borderRadius: '4px',
          backgroundColor: color,
          border: `1px solid ${isActive ? color : '#444444'}`,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          transition: 'all 0.3s ease',
          boxShadow: isActive ? `0 0 8px ${color}` : 'none',
          '&:hover': {
            transform: 'scale(1.1)',
          },
        }}
      >
        <Typography
          variant="caption"
          sx={{
            color: isActive ? '#000' : '#666',
            fontWeight: 600,
            fontSize: '0.65rem',
          }}
        >
          {index + 1}
        </Typography>
      </Box>
    </Tooltip>
  );
}

/**
 * Fold visualization grid
 */
function FoldVisualization({ current, total }) {
  const folds = Array.from({ length: total }, (_, i) => ({
    index: i,
    status: i < current ? 'completed' : 'pending',
    isCurrent: i === current,
  }));

  return (
    <Box
      sx={{
        display: 'flex',
        flexWrap: 'wrap',
        gap: 0.5,
        mt: 1,
        p: 1.5,
        backgroundColor: '#0a0a0a',
        borderRadius: '8px',
        border: '1px solid #222222',
      }}
    >
      {folds.map((fold) => (
        <FoldBox
          key={fold.index}
          index={fold.index}
          status={fold.status}
          isCurrent={fold.isCurrent}
        />
      ))}
    </Box>
  );
}

/**
 * Main OptimizationProgress component
 */
export default function OptimizationProgress({
  tid,
  progress,
  status,
  isConnected,
  isComplete,
  error,
}) {
  const [expanded, setExpanded] = useState(true);
  const [isCancelling, setIsCancelling] = useState(false);
  const { notify } = useNotification();

  const statusConfig = STATUS_CONFIG[status] || STATUS_CONFIG.pending;
  const StatusIcon = statusConfig.icon;
  const showFolds = progress.total > 1;
  const isRunning = status === 'running' || status === 'pending';

  /**
   * Cancel the running optimization
   */
  const handleCancel = async () => {
    if (!tid || isCancelling) return;

    setIsCancelling(true);
    try {
      const apiUrl = process.env.REACT_APP_REST_API_V2_URL || process.env.REACT_APP_REST_API_URL;
      const response = await fetch(`${apiUrl}/api/v2/optimization/${tid}/cancel`, {
        method: 'POST',
      });

      if (!response.ok) {
        throw new Error(`Cancel failed: HTTP ${response.status}`);
      }

      notify('Optimization cancelled', 'info');
    } catch (err) {
      notify(`Failed to cancel: ${err.message}`, 'error');
    } finally {
      setIsCancelling(false);
    }
  };

  return (
    <Box
      sx={{
        mt: 2,
        p: 2,
        backgroundColor: '#111111',
        border: `1px solid ${statusConfig.borderColor}`,
        borderRadius: '12px',
        transition: 'all 0.3s ease',
        boxShadow: isRunning ? `0 0 15px ${statusConfig.borderColor}33` : 'none',
      }}
    >
      {/* Header row with status and connection chips */}
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          mb: 1.5,
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          {/* Status chip */}
          <Chip
            icon={<StatusIcon sx={{ fontSize: 16 }} />}
            label={statusConfig.label}
            size="small"
            color={statusConfig.color}
            variant="outlined"
            sx={{
              backgroundColor: statusConfig.bgColor,
              borderColor: statusConfig.borderColor,
              '& .MuiChip-icon': {
                color: statusConfig.borderColor,
              },
            }}
          />

          {/* Connection indicator */}
          <Chip
            icon={isConnected ? <WifiIcon sx={{ fontSize: 14 }} /> : <SyncIcon sx={{ fontSize: 14 }} />}
            label={isConnected ? 'Live' : 'Polling'}
            size="small"
            variant="outlined"
            sx={{
              backgroundColor: isConnected
                ? 'rgba(16, 185, 129, 0.1)'
                : 'rgba(6, 182, 212, 0.1)',
              borderColor: isConnected ? '#10b981' : '#06B6D4',
              '& .MuiChip-icon': {
                color: isConnected ? '#10b981' : '#06B6D4',
              },
              '& .MuiChip-label': {
                color: isConnected ? '#10b981' : '#06B6D4',
              },
            }}
          />
        </Box>

        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
          {/* Stop button */}
          {isRunning && (
            <Tooltip title="Stop optimization">
              <IconButton
                size="small"
                onClick={handleCancel}
                disabled={isCancelling}
                sx={{
                  color: '#ef4444',
                  '&:hover': {
                    backgroundColor: 'rgba(239, 68, 68, 0.1)',
                  },
                }}
              >
                <StopIcon fontSize="small" />
              </IconButton>
            </Tooltip>
          )}

          {/* Expand/collapse toggle */}
          {showFolds && (
            <Tooltip title={expanded ? 'Collapse details' : 'Expand details'}>
              <IconButton
                size="small"
                onClick={() => setExpanded(!expanded)}
                sx={{ color: '#9ca3af' }}
              >
                {expanded ? <ExpandLessIcon /> : <ExpandMoreIcon />}
              </IconButton>
            </Tooltip>
          )}
        </Box>
      </Box>

      {/* Progress bar with percentage */}
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
        <Box sx={{ flexGrow: 1 }}>
          <LinearProgress
            variant="determinate"
            value={progress.percent}
            sx={{
              height: 8,
              borderRadius: '4px',
              backgroundColor: '#222222',
              '& .MuiLinearProgress-bar': {
                backgroundColor: statusConfig.borderColor,
                borderRadius: '4px',
                boxShadow: isRunning ? `0 0 10px ${statusConfig.borderColor}` : 'none',
              },
            }}
          />
        </Box>
        <Typography
          variant="body2"
          sx={{
            minWidth: 45,
            color: statusConfig.borderColor,
            fontFamily: '"IBM Plex Mono", monospace',
            fontWeight: 600,
          }}
        >
          {Math.round(progress.percent)}%
        </Typography>
      </Box>

      {/* Progress message */}
      {progress.message && (
        <Typography
          variant="caption"
          sx={{
            display: 'block',
            mt: 1,
            color: 'text.secondary',
            fontFamily: '"IBM Plex Mono", monospace',
          }}
        >
          {progress.message}
        </Typography>
      )}

      {/* Fold count indicator */}
      {showFolds && (
        <Typography
          variant="caption"
          sx={{
            display: 'block',
            mt: 0.5,
            color: 'text.secondary',
          }}
        >
          Fold {progress.current} of {progress.total}
        </Typography>
      )}

      {/* Error display */}
      {error && (
        <Typography
          variant="body2"
          sx={{
            mt: 1,
            color: 'error.main',
            fontFamily: '"IBM Plex Mono", monospace',
          }}
        >
          {error}
        </Typography>
      )}

      {/* Expandable fold visualization */}
      {showFolds && (
        <Collapse in={expanded}>
          <FoldVisualization current={progress.current} total={progress.total} />
        </Collapse>
      )}
    </Box>
  );
}
