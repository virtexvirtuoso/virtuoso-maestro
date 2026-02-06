// OptunaVisualization - Native visualization for Optuna optimization results
import React, { useState, useEffect, useCallback } from 'react';
import Box from '@mui/material/Box';
import Paper from '@mui/material/Paper';
import Typography from '@mui/material/Typography';
import Grid from '@mui/material/Grid';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableContainer from '@mui/material/TableContainer';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import Chip from '@mui/material/Chip';
import CircularProgress from '@mui/material/CircularProgress';
import Alert from '@mui/material/Alert';
import ScienceIcon from '@mui/icons-material/Science';
import EmojiEventsIcon from '@mui/icons-material/EmojiEvents';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Cell,
} from 'recharts';

const AMBER = '#fbbf24';
const GREEN = '#10b981';
const CYAN = '#06B6D4';

// Custom tooltip for the scatter chart
const CustomTooltip = ({ active, payload }) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    return (
      <Paper
        sx={{
          p: 1.5,
          backgroundColor: 'background.paper',
          border: '1px solid',
          borderColor: data.isBest ? GREEN : 'divider',
        }}
      >
        <Typography variant="body2" sx={{ fontWeight: 600, color: data.isBest ? GREEN : 'text.primary' }}>
          Trial #{data.trial}
          {data.isBest && ' (Best)'}
        </Typography>
        <Typography variant="body2" sx={{ color: 'text.secondary', mt: 0.5 }}>
          Objective: {data.value?.toFixed(6) || 'N/A'}
        </Typography>
        {data.state && (
          <Typography variant="body2" sx={{ color: 'text.secondary' }}>
            State: {data.state}
          </Typography>
        )}
      </Paper>
    );
  }
  return null;
};

export default function OptunaVisualization({ studyId, studyData }) {
  const [trials, setTrials] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [studyInfo, setStudyInfo] = useState(null);

  const processStudyData = useCallback((data, id) => {
    // Handle both formats: direct trial array or nested structure
    const trialList = Array.isArray(data) ? data : data.trials || [];

    // Find best trial
    const completedTrials = trialList.filter(
      (t) => t.state === 'COMPLETE' && t.value != null
    );
    const bestValue = completedTrials.length > 0
      ? Math.max(...completedTrials.map((t) => t.value))
      : null;

    // Process trials for visualization
    const processedTrials = trialList.map((trial, index) => ({
      trial: trial.number ?? index,
      value: trial.value,
      state: trial.state || 'COMPLETE',
      params: trial.params || {},
      isBest: trial.value === bestValue && trial.state === 'COMPLETE',
    }));

    setTrials(processedTrials);
    setStudyInfo({
      name: data.study_name || data.name || `Study ${id}`,
      totalTrials: trialList.length,
      bestValue: bestValue,
      bestTrial: processedTrials.find((t) => t.isBest),
    });
  }, []);

  useEffect(() => {
    // If studyData is passed directly, use it
    if (studyData) {
      processStudyData(studyData, studyId);
      return;
    }

    // Otherwise fetch from API if studyId is provided
    if (!studyId) return;

    const fetchTrials = async () => {
      setLoading(true);
      setError(null);
      try {
        const apiUrl = process.env.REACT_APP_REST_API_V2_URL || process.env.REACT_APP_REST_API_URL;
        const response = await fetch(`${apiUrl}/api/v2/optuna/studies/${studyId}/trials`);
        if (!response.ok) {
          throw new Error(`Failed to fetch trials: ${response.statusText}`);
        }
        const data = await response.json();
        processStudyData(data, studyId);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchTrials();
  }, [studyId, studyData, processStudyData]);

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: 200 }}>
        <CircularProgress sx={{ color: AMBER }} />
      </Box>
    );
  }

  if (error) {
    return (
      <Alert severity="error" sx={{ mb: 2 }}>
        {error}
      </Alert>
    );
  }

  if (!studyInfo || trials.length === 0) {
    return (
      <Alert severity="info" icon={<ScienceIcon />}>
        No optimization trial data available. Run an Optuna optimization study to see results here.
      </Alert>
    );
  }

  // Prepare scatter chart data
  const chartData = trials
    .filter((t) => t.value != null)
    .map((t) => ({
      ...t,
      x: t.trial,
      y: t.value,
    }));

  return (
    <Box sx={{ width: '100%' }}>
      {/* Study Info Card */}
      <Paper
        sx={{
          p: 2,
          mb: 3,
          display: 'flex',
          alignItems: 'center',
          gap: 3,
          flexWrap: 'wrap',
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <ScienceIcon sx={{ color: CYAN }} />
          <Box>
            <Typography variant="caption" sx={{ color: 'text.secondary' }}>
              Study Name
            </Typography>
            <Typography variant="body1" sx={{ fontWeight: 600 }}>
              {studyInfo.name}
            </Typography>
          </Box>
        </Box>

        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <TrendingUpIcon sx={{ color: AMBER }} />
          <Box>
            <Typography variant="caption" sx={{ color: 'text.secondary' }}>
              Total Trials
            </Typography>
            <Typography variant="body1" sx={{ fontWeight: 600 }}>
              {studyInfo.totalTrials}
            </Typography>
          </Box>
        </Box>

        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <EmojiEventsIcon sx={{ color: GREEN }} />
          <Box>
            <Typography variant="caption" sx={{ color: 'text.secondary' }}>
              Best Value
            </Typography>
            <Typography variant="body1" sx={{ fontWeight: 600, color: GREEN }}>
              {studyInfo.bestValue?.toFixed(6) || 'N/A'}
            </Typography>
          </Box>
        </Box>

        {studyInfo.bestTrial && (
          <Chip
            label={`Best: Trial #${studyInfo.bestTrial.trial}`}
            sx={{
              backgroundColor: 'rgba(16, 185, 129, 0.15)',
              color: GREEN,
              fontWeight: 600,
              border: `1px solid ${GREEN}`,
            }}
          />
        )}
      </Paper>

      <Grid container spacing={3}>
        {/* Optimization History Scatter Chart */}
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" sx={{ mb: 2, fontFamily: '"IBM Plex Mono", monospace' }}>
              Optimization History
            </Typography>
            <ResponsiveContainer width="100%" height={300}>
              <ScatterChart margin={{ top: 20, right: 30, bottom: 20, left: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#222" />
                <XAxis
                  dataKey="x"
                  name="Trial"
                  type="number"
                  tick={{ fill: '#9ca3af', fontSize: 12 }}
                  axisLine={{ stroke: '#333' }}
                  tickLine={{ stroke: '#333' }}
                  label={{
                    value: 'Trial Number',
                    position: 'bottom',
                    fill: '#9ca3af',
                    fontSize: 12,
                  }}
                />
                <YAxis
                  dataKey="y"
                  name="Objective"
                  tick={{ fill: '#9ca3af', fontSize: 12 }}
                  axisLine={{ stroke: '#333' }}
                  tickLine={{ stroke: '#333' }}
                  label={{
                    value: 'Objective Value',
                    angle: -90,
                    position: 'insideLeft',
                    fill: '#9ca3af',
                    fontSize: 12,
                  }}
                />
                <Tooltip content={<CustomTooltip />} />
                {studyInfo.bestValue != null && (
                  <ReferenceLine
                    y={studyInfo.bestValue}
                    stroke={GREEN}
                    strokeDasharray="5 5"
                    label={{
                      value: 'Best',
                      fill: GREEN,
                      fontSize: 10,
                      position: 'right',
                    }}
                  />
                )}
                <Scatter name="Trials" data={chartData} fill={AMBER}>
                  {chartData.map((entry, index) => (
                    <Cell
                      key={`cell-${index}`}
                      fill={entry.isBest ? GREEN : AMBER}
                      stroke={entry.isBest ? GREEN : AMBER}
                      strokeWidth={entry.isBest ? 3 : 1}
                      r={entry.isBest ? 8 : 5}
                    />
                  ))}
                </Scatter>
              </ScatterChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>

        {/* Best Parameters Table */}
        {studyInfo.bestTrial && Object.keys(studyInfo.bestTrial.params).length > 0 && (
          <Grid item xs={12}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" sx={{ mb: 2, fontFamily: '"IBM Plex Mono", monospace' }}>
                Best Parameters
              </Typography>
              <TableContainer>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Parameter</TableCell>
                      <TableCell align="right">Value</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {Object.entries(studyInfo.bestTrial.params).map(([param, value]) => (
                      <TableRow key={param} hover>
                        <TableCell
                          sx={{
                            fontFamily: '"IBM Plex Mono", monospace',
                            fontWeight: 500,
                          }}
                        >
                          {param}
                        </TableCell>
                        <TableCell
                          align="right"
                          sx={{
                            fontFamily: '"IBM Plex Mono", monospace',
                            color: AMBER,
                          }}
                        >
                          {typeof value === 'number' ? value.toFixed(6) : String(value)}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </Paper>
          </Grid>
        )}
      </Grid>
    </Box>
  );
}
