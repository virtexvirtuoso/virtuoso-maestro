import React, { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import Grid from '@mui/material/Grid';
import Paper from '@mui/material/Paper';
import Title from './Title';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableRow from '@mui/material/TableRow';
import TableHead from '@mui/material/TableHead';
import Table from '@mui/material/Table';
import TableContainer from '@mui/material/TableContainer';
import IconButton from '@mui/material/IconButton';
import ViewListIcon from '@mui/icons-material/ViewList';
import { TreeView } from '@mui/x-tree-view/TreeView';
import { TreeItem } from '@mui/x-tree-view/TreeItem';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ChevronRightIcon from '@mui/icons-material/ChevronRight';
import DeleteIcon from '@mui/icons-material/Delete';

function PerformedTests({ rowElement }) {
  const tests = Object.keys(rowElement);

  return (
    <TreeView
      defaultCollapseIcon={<ExpandMoreIcon />}
      defaultExpandIcon={<ChevronRightIcon />}
    >
      {tests
        .filter((t) => rowElement[t].length > 0)
        .map((t, i) => (
          <TreeItem nodeId={i.toString()} key={t} label={t}>
            {rowElement[t].map((test, j) => (
              <TreeItem
                nodeId={`params-${i}-${j}`}
                key={`params-${i}-${j}`}
                label={'Parameters ' + (test['num_split'] !== null ? test['num_split'] : '')}
              >
                {Object.keys(test['parameters']).map((p) => (
                  <TreeItem
                    nodeId={`params-${i}-${j}-${p}`}
                    key={`params-${i}-${j}-${p}`}
                    label={p + ': ' + test['parameters'][p]}
                  />
                ))}
              </TreeItem>
            ))}
          </TreeItem>
        ))}
    </TreeView>
  );
}

export default function OptimizationResults() {
  const navigate = useNavigate();
  const [results, setResults] = useState([]);

  const fetchResults = useCallback(() => {
    fetch(`${process.env.REACT_APP_REST_API_URL}/optimization/results`)
      .then((response) => response.json())
      .then((data) => {
        data.sort((a, b) =>
          a['creation_time'] < b['creation_time']
            ? 1
            : a['creation_time'] > b['creation_time']
            ? -1
            : 0
        );
        setResults(data);
      })
      .catch((error) => console.log(error));
  }, []);

  useEffect(() => {
    fetchResults();
  }, [fetchResults]);

  const handleComparePage = useCallback(
    (tid) => {
      navigate(`/evaluate/${tid}`);
    },
    [navigate]
  );

  const handleDeleteOptResult = useCallback(
    (tid) => {
      fetch(`${process.env.REACT_APP_REST_API_URL}/optimization/results/${tid}`, {
        method: 'DELETE',
      }).then(() => fetchResults());
    },
    [fetchResults]
  );

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
          <Title>Results</Title>
          <TableContainer>
            <Table aria-label="Results Table" size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Name</TableCell>
                  <TableCell>Creation Time</TableCell>
                  <TableCell>Strategy</TableCell>
                  <TableCell>Provider</TableCell>
                  <TableCell>Symbol</TableCell>
                  <TableCell>Timeframe</TableCell>
                  <TableCell>Tests</TableCell>
                  <TableCell />
                </TableRow>
              </TableHead>
              <TableBody>
                {results.map((row) => (
                  <TableRow key={row['tid']} hover>
                    <TableCell>{row['test_name']}</TableCell>
                    <TableCell>{new Date(row['creation_time']).toLocaleString()}</TableCell>
                    <TableCell>{row['strategy']}</TableCell>
                    <TableCell>{row['provider']}</TableCell>
                    <TableCell>{row['symbol']}</TableCell>
                    <TableCell>{row['timeframe']}</TableCell>
                    <TableCell>
                      <PerformedTests rowElement={row['optimizations']} />
                    </TableCell>
                    <TableCell>
                      <IconButton onClick={() => handleComparePage(row['tid'])}>
                        <ViewListIcon />
                      </IconButton>
                      <IconButton onClick={() => handleDeleteOptResult(row['tid'])}>
                        <DeleteIcon />
                      </IconButton>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </Paper>
      </Grid>
    </Grid>
  );
}
