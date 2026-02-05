import React from 'react';
import Grid from '@mui/material/Grid';
import Paper from '@mui/material/Paper';
import CandleStickChart from './CandleStickChart';
import OptimizationForm from './OptimizationForm';

export default function Home() {
  return (
    <Grid container spacing={3}>
      <Grid item xs={12}>
        <Paper
          sx={{
            p: 2,
            display: 'flex',
            overflow: 'auto',
            flexDirection: 'column',
          }}
        >
          <CandleStickChart />
        </Paper>
      </Grid>
      <Grid item xs={12}>
        <Paper
          sx={{
            p: 2,
            display: 'flex',
            overflow: 'auto',
            flexDirection: 'column',
          }}
        >
          <OptimizationForm />
        </Paper>
      </Grid>
    </Grid>
  );
}
