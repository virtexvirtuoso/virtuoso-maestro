import React, { useState, useEffect, useCallback, useMemo } from 'react';
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
import TablePagination from '@mui/material/TablePagination';
import TableSortLabel from '@mui/material/TableSortLabel';
import IconButton from '@mui/material/IconButton';
import ViewListIcon from '@mui/icons-material/ViewList';
import { TreeView } from '@mui/x-tree-view/TreeView';
import { TreeItem } from '@mui/x-tree-view/TreeItem';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ChevronRightIcon from '@mui/icons-material/ChevronRight';
import DeleteIcon from '@mui/icons-material/Delete';
import EmptyState from './EmptyState';
import ScienceIcon from '@mui/icons-material/Science';
import SearchOffIcon from '@mui/icons-material/SearchOff';
import Chip from '@mui/material/Chip';
import Autocomplete from '@mui/material/Autocomplete';
import TextField from '@mui/material/TextField';
import Box from '@mui/material/Box';

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

// Status chip component for consistent styling
function StatusChip({ status }) {
  const statusConfig = {
    completed: { color: 'success', label: 'Completed' },
    failed: { color: 'error', label: 'Failed' },
    running: { color: 'primary', label: 'Running' },
    pending: { color: 'default', label: 'Pending' },
  };

  const config = statusConfig[status?.toLowerCase()] || statusConfig.pending;

  return (
    <Chip
      label={config.label}
      color={config.color}
      size="small"
      sx={{ fontWeight: 500 }}
    />
  );
}

// Format Sharpe ratio to 2 decimal places
function formatSharpe(value) {
  if (value === null || value === undefined || isNaN(value)) {
    return '—';
  }
  return Number(value).toFixed(2);
}

// Extract best Sharpe from optimizations
function extractBestSharpe(optimizations) {
  if (!optimizations) return null;

  let bestSharpe = null;
  Object.values(optimizations).forEach((tests) => {
    if (Array.isArray(tests)) {
      tests.forEach((test) => {
        if (test?.metrics?.sharpe_ratio !== undefined) {
          const sharpe = Number(test.metrics.sharpe_ratio);
          if (bestSharpe === null || sharpe > bestSharpe) {
            bestSharpe = sharpe;
          }
        }
      });
    }
  });
  return bestSharpe;
}

// Column configuration for sortable headers
const columns = [
  { id: 'test_name', label: 'Name', sortable: false },
  { id: 'creation_time', label: 'Date', sortable: true },
  { id: 'strategy', label: 'Strategy', sortable: true },
  { id: 'provider', label: 'Provider', sortable: false },
  { id: 'symbol', label: 'Symbol', sortable: true },
  { id: 'timeframe', label: 'Timeframe', sortable: false },
  { id: 'sharpe', label: 'Sharpe', sortable: true },
  { id: 'status', label: 'Status', sortable: true },
  { id: 'tests', label: 'Tests', sortable: false },
  { id: 'actions', label: '', sortable: false },
];

export default function OptimizationResults() {
  const navigate = useNavigate();
  const [results, setResults] = useState([]);

  // Sorting state
  const [orderBy, setOrderBy] = useState('creation_time');
  const [order, setOrder] = useState('desc');

  // Filter state
  const [strategyFilter, setStrategyFilter] = useState(null);

  // Pagination state
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);

  const fetchResults = useCallback(() => {
    fetch(`${process.env.REACT_APP_REST_API_URL}/optimization/results`)
      .then((response) => response.json())
      .then((data) => {
        // Enhance data with computed fields
        const enhancedData = data.map((row) => ({
          ...row,
          sharpe: extractBestSharpe(row.optimizations),
          status: row.status || 'completed', // Default to completed if not specified
        }));
        setResults(enhancedData);
      })
      .catch((error) => console.log(error));
  }, []);

  useEffect(() => {
    fetchResults();
  }, [fetchResults]);

  // Get unique strategies for filter options
  const strategyOptions = useMemo(() => {
    const strategies = [...new Set(results.map((r) => r.strategy))];
    return strategies.filter(Boolean).sort();
  }, [results]);

  // Handle sort request
  const handleRequestSort = useCallback((property) => {
    const isAsc = orderBy === property && order === 'asc';
    setOrder(isAsc ? 'desc' : 'asc');
    setOrderBy(property);
  }, [orderBy, order]);

  // Filter and sort results
  const processedResults = useMemo(() => {
    let filtered = results;

    // Apply strategy filter
    if (strategyFilter) {
      filtered = filtered.filter((r) => r.strategy === strategyFilter);
    }

    // Sort results
    const comparator = (a, b) => {
      let aVal = a[orderBy];
      let bVal = b[orderBy];

      // Handle special cases
      if (orderBy === 'creation_time') {
        aVal = new Date(aVal).getTime();
        bVal = new Date(bVal).getTime();
      } else if (orderBy === 'sharpe') {
        aVal = aVal ?? -Infinity;
        bVal = bVal ?? -Infinity;
      }

      // Compare
      if (aVal < bVal) return order === 'asc' ? -1 : 1;
      if (aVal > bVal) return order === 'asc' ? 1 : -1;
      return 0;
    };

    return [...filtered].sort(comparator);
  }, [results, strategyFilter, orderBy, order]);

  // Get paginated results
  const paginatedResults = useMemo(() => {
    const startIndex = page * rowsPerPage;
    return processedResults.slice(startIndex, startIndex + rowsPerPage);
  }, [processedResults, page, rowsPerPage]);

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

  const handleRunTest = useCallback(() => {
    navigate('/');
  }, [navigate]);

  const handleChangePage = useCallback((event, newPage) => {
    setPage(newPage);
  }, []);

  const handleChangeRowsPerPage = useCallback((event) => {
    setRowsPerPage(parseInt(event.target.value, 10));
    setPage(0);
  }, []);

  const handleStrategyFilterChange = useCallback((event, newValue) => {
    setStrategyFilter(newValue);
    setPage(0); // Reset to first page when filter changes
  }, []);

  // Show initial empty state if no results at all
  if (results.length === 0) {
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
            <EmptyState
              icon={ScienceIcon}
              title="No Tests Yet"
              description="Run your first optimization test to see results here. Configure a strategy and backtest parameters to get started."
              actionLabel="Run Test"
              onAction={handleRunTest}
            />
          </Paper>
        </Grid>
      </Grid>
    );
  }

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
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Title>Results</Title>
            <Autocomplete
              value={strategyFilter}
              onChange={handleStrategyFilterChange}
              options={strategyOptions}
              sx={{ width: 250 }}
              renderInput={(params) => (
                <TextField
                  {...params}
                  label="Filter by Strategy"
                  size="small"
                  placeholder="All strategies"
                />
              )}
            />
          </Box>

          {processedResults.length === 0 ? (
            <EmptyState
              icon={SearchOffIcon}
              title="No Matching Results"
              description={`No results found for strategy "${strategyFilter}". Try selecting a different strategy or clear the filter.`}
              actionLabel="Clear Filter"
              onAction={() => setStrategyFilter(null)}
            />
          ) : (
            <>
              <TableContainer>
                <Table aria-label="Results Table" size="small">
                  <TableHead>
                    <TableRow>
                      {columns.map((column) => (
                        <TableCell key={column.id}>
                          {column.sortable ? (
                            <TableSortLabel
                              active={orderBy === column.id}
                              direction={orderBy === column.id ? order : 'asc'}
                              onClick={() => handleRequestSort(column.id)}
                            >
                              {column.label}
                            </TableSortLabel>
                          ) : (
                            column.label
                          )}
                        </TableCell>
                      ))}
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {paginatedResults.map((row) => (
                      <TableRow key={row['tid']} hover>
                        <TableCell>{row['test_name']}</TableCell>
                        <TableCell>{new Date(row['creation_time']).toLocaleString()}</TableCell>
                        <TableCell>{row['strategy']}</TableCell>
                        <TableCell>{row['provider']}</TableCell>
                        <TableCell>{row['symbol']}</TableCell>
                        <TableCell>{row['timeframe']}</TableCell>
                        <TableCell sx={{ fontFamily: 'monospace' }}>
                          {formatSharpe(row['sharpe'])}
                        </TableCell>
                        <TableCell>
                          <StatusChip status={row['status']} />
                        </TableCell>
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
              <TablePagination
                rowsPerPageOptions={[10, 25, 50]}
                component="div"
                count={processedResults.length}
                rowsPerPage={rowsPerPage}
                page={page}
                onPageChange={handleChangePage}
                onRowsPerPageChange={handleChangeRowsPerPage}
              />
            </>
          )}
        </Paper>
      </Grid>
    </Grid>
  );
}
