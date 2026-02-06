// Walk-Forward Timeline - Horizontal bar visualization of train/test periods across splits
import React, { useMemo } from 'react';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Tooltip from '@mui/material/Tooltip';
import { useTheme } from '@mui/material/styles';

/**
 * WalkForwardTimeline - Displays train (dark primary) and test (success) periods
 * for each walk-forward optimization split as horizontal bars.
 *
 * @param {Object} props
 * @param {Array} props.splits - Array of split objects with train_start, train_end, test_start, test_end
 * @param {string} props.startDate - Overall start date of the data
 * @param {string} props.endDate - Overall end date of the data
 */
export default function WalkForwardTimeline({ splits, startDate, endDate }) {
  const theme = useTheme();

  // Parse and compute relative positions for the timeline
  const timelineData = useMemo(() => {
    if (!splits || splits.length === 0) return null;

    const start = new Date(startDate).getTime();
    const end = new Date(endDate).getTime();
    const totalDuration = end - start;

    if (totalDuration <= 0) return null;

    return splits
      .sort((a, b) => (a.num_split < b.num_split ? -1 : 1))
      .map((split, index) => {
        const trainStart = new Date(split.train_start || split.start_date).getTime();
        const trainEnd = new Date(split.train_end || split.end_date).getTime();
        const testStart = new Date(split.test_start || trainEnd).getTime();
        const testEnd = new Date(split.test_end || split.end_date).getTime();

        return {
          splitNum: split.num_split ?? index,
          trainStartPct: ((trainStart - start) / totalDuration) * 100,
          trainWidthPct: ((trainEnd - trainStart) / totalDuration) * 100,
          testStartPct: ((testStart - start) / totalDuration) * 100,
          testWidthPct: ((testEnd - testStart) / totalDuration) * 100,
          trainStartDate: new Date(trainStart).toLocaleDateString(),
          trainEndDate: new Date(trainEnd).toLocaleDateString(),
          testStartDate: new Date(testStart).toLocaleDateString(),
          testEndDate: new Date(testEnd).toLocaleDateString(),
        };
      });
  }, [splits, startDate, endDate]);

  if (!timelineData || timelineData.length === 0) {
    return (
      <Typography variant="body2" color="text.secondary">
        No walk-forward splits available
      </Typography>
    );
  }

  const formatDate = (dateStr) => {
    return new Date(dateStr).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
    });
  };

  return (
    <Box sx={{ width: '100%', py: 2 }}>
      {/* Timeline rows */}
      <Box sx={{ position: 'relative' }}>
        {timelineData.map((split, index) => (
          <Box
            key={split.splitNum}
            sx={{
              display: 'flex',
              alignItems: 'center',
              mb: 1,
              height: 28,
            }}
          >
            {/* Split label */}
            <Typography
              variant="caption"
              sx={{
                width: 60,
                flexShrink: 0,
                fontFamily: '"IBM Plex Mono", monospace',
                color: 'text.secondary',
              }}
            >
              Split {split.splitNum}
            </Typography>

            {/* Timeline bar container */}
            <Box
              sx={{
                flex: 1,
                height: 20,
                position: 'relative',
                backgroundColor: theme.palette.background.paper,
                borderRadius: 1,
                border: `1px solid ${theme.palette.divider}`,
                overflow: 'hidden',
              }}
            >
              {/* Train period bar */}
              <Tooltip
                title={
                  <Box>
                    <Typography variant="caption" sx={{ fontWeight: 600 }}>
                      Training Period
                    </Typography>
                    <br />
                    <Typography variant="caption">
                      {split.trainStartDate} → {split.trainEndDate}
                    </Typography>
                  </Box>
                }
                arrow
                placement="top"
              >
                <Box
                  sx={{
                    position: 'absolute',
                    left: `${split.trainStartPct}%`,
                    width: `${split.trainWidthPct}%`,
                    height: '100%',
                    backgroundColor: theme.palette.primary.dark,
                    opacity: 0.85,
                    cursor: 'pointer',
                    transition: 'opacity 0.2s ease',
                    '&:hover': {
                      opacity: 1,
                    },
                  }}
                />
              </Tooltip>

              {/* Test period bar */}
              <Tooltip
                title={
                  <Box>
                    <Typography variant="caption" sx={{ fontWeight: 600 }}>
                      Test Period (Out-of-Sample)
                    </Typography>
                    <br />
                    <Typography variant="caption">
                      {split.testStartDate} → {split.testEndDate}
                    </Typography>
                  </Box>
                }
                arrow
                placement="top"
              >
                <Box
                  sx={{
                    position: 'absolute',
                    left: `${split.testStartPct}%`,
                    width: `${split.testWidthPct}%`,
                    height: '100%',
                    backgroundColor: theme.palette.success.main,
                    opacity: 0.85,
                    cursor: 'pointer',
                    transition: 'opacity 0.2s ease',
                    '&:hover': {
                      opacity: 1,
                    },
                  }}
                />
              </Tooltip>
            </Box>
          </Box>
        ))}
      </Box>

      {/* Date labels at bottom */}
      <Box
        sx={{
          display: 'flex',
          justifyContent: 'space-between',
          mt: 2,
          pl: '60px', // Align with timeline bars
        }}
      >
        <Typography
          variant="caption"
          sx={{
            fontFamily: '"IBM Plex Mono", monospace',
            color: 'text.secondary',
          }}
        >
          {formatDate(startDate)}
        </Typography>
        <Typography
          variant="caption"
          sx={{
            fontFamily: '"IBM Plex Mono", monospace',
            color: 'text.secondary',
          }}
        >
          {formatDate(endDate)}
        </Typography>
      </Box>

      {/* Legend */}
      <Box
        sx={{
          display: 'flex',
          gap: 3,
          mt: 2,
          pl: '60px',
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Box
            sx={{
              width: 16,
              height: 12,
              backgroundColor: theme.palette.primary.dark,
              borderRadius: 0.5,
            }}
          />
          <Typography variant="caption" color="text.secondary">
            Training (In-Sample)
          </Typography>
        </Box>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Box
            sx={{
              width: 16,
              height: 12,
              backgroundColor: theme.palette.success.main,
              borderRadius: 0.5,
            }}
          />
          <Typography variant="caption" color="text.secondary">
            Test (Out-of-Sample)
          </Typography>
        </Box>
      </Box>
    </Box>
  );
}
