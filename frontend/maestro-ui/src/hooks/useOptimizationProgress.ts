/**
 * useOptimizationProgress - WebSocket progress hook with polling fallback
 *
 * Provides real-time optimization progress updates via WebSocket.
 * Automatically falls back to HTTP polling if WebSocket fails.
 */

import { useState, useEffect, useCallback, useRef } from 'react';

/**
 * Progress state interface for optimization jobs
 */
export interface Progress {
  current: number;
  total: number;
  percent: number;
  message: string;
}

/**
 * WebSocket message format from FastAPI backend
 */
interface WsProgressMessage {
  tid: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'connected';
  progress_pct: number;
  current_fold?: number;
  total_folds?: number;
  message?: string;
  type?: 'keepalive' | 'progress';
}

/**
 * API polling response format (same as WsProgressMessage)
 */
type ApiProgressResponse = WsProgressMessage;

/**
 * Hook return type
 */
interface UseOptimizationProgressReturn {
  progress: Progress;
  isConnected: boolean;
  isComplete: boolean;
  error: string | null;
  status: string;
}

const POLLING_INTERVAL_MS = 2000;
const WS_RECONNECT_DELAY_MS = 1000;

/**
 * Hook for tracking optimization progress with WebSocket and polling fallback
 *
 * @param tid - Task ID for the optimization job
 * @param enabled - Whether to start tracking (default: true)
 * @returns Progress state, connection status, and any errors
 *
 * @example
 * const { progress, isComplete, error } = useOptimizationProgress(taskId);
 * return <ProgressBar value={progress.percent} />;
 */
export function useOptimizationProgress(
  tid: string | null,
  enabled: boolean = true
): UseOptimizationProgressReturn {
  const [progress, setProgress] = useState<Progress>({
    current: 0,
    total: 100,
    percent: 0,
    message: '',
  });
  const [isConnected, setIsConnected] = useState(false);
  const [isComplete, setIsComplete] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<string>('pending');

  // Refs to track WebSocket and polling state
  const wsRef = useRef<WebSocket | null>(null);
  const pollingIntervalRef = useRef<number | null>(null);
  const usingPollingRef = useRef(false);
  const mountedRef = useRef(true);

  /**
   * Update progress state from WebSocket or polling message
   */
  const handleProgressUpdate = useCallback((data: WsProgressMessage) => {
    if (!mountedRef.current) return;

    // Skip keepalive messages
    if (data.type === 'keepalive') return;

    const newPercent = data.progress_pct ?? 0;
    const currentFold = data.current_fold ?? 0;
    const totalFolds = data.total_folds ?? 100;

    setProgress({
      current: currentFold,
      total: totalFolds,
      percent: newPercent,
      message: data.message ?? '',
    });

    setStatus(data.status);

    // Check for completion
    if (data.status === 'completed' || data.status === 'failed' || newPercent >= 100) {
      setIsComplete(true);
      if (data.status === 'failed') {
        setError(data.message ?? 'Optimization failed');
      }
    }
  }, []);

  /**
   * Start HTTP polling as fallback
   */
  const startPolling = useCallback(() => {
    if (!tid || pollingIntervalRef.current || !mountedRef.current) return;

    usingPollingRef.current = true;

    const poll = async () => {
      if (!mountedRef.current) return;

      try {
        const apiUrl = process.env.REACT_APP_REST_API_V2_URL || '';
        const response = await fetch(`${apiUrl}/api/v2/optimization/${tid}/progress`);

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }

        const data: ApiProgressResponse = await response.json();
        handleProgressUpdate(data);

        // Stop polling on completion
        if (data.progress_pct >= 100 || data.status === 'completed' || data.status === 'failed') {
          if (pollingIntervalRef.current) {
            window.clearInterval(pollingIntervalRef.current);
            pollingIntervalRef.current = null;
          }
        }
      } catch (err) {
        console.error('Polling error:', err);
      }
    };

    // Initial poll
    poll();

    // Start interval
    pollingIntervalRef.current = window.setInterval(poll, POLLING_INTERVAL_MS);
  }, [tid, handleProgressUpdate]);

  /**
   * Connect to WebSocket for real-time updates
   */
  const connectWebSocket = useCallback(() => {
    if (!tid || !mountedRef.current) return;

    // Build WebSocket URL
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = process.env.REACT_APP_WS_HOST || window.location.host;
    const wsUrl = `${protocol}//${host}/api/v2/optimization/${tid}/ws`;

    try {
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        if (!mountedRef.current) return;
        setIsConnected(true);
        setError(null);
      };

      ws.onmessage = (event) => {
        if (!mountedRef.current) return;

        try {
          const data: WsProgressMessage = JSON.parse(event.data);
          handleProgressUpdate(data);
        } catch (parseErr) {
          console.error('Failed to parse WebSocket message:', parseErr);
        }
      };

      ws.onerror = (event) => {
        console.warn('WebSocket error, falling back to polling:', event);
        if (!mountedRef.current) return;

        setIsConnected(false);

        // Close WebSocket and switch to polling
        ws.close();
        wsRef.current = null;

        // Start polling fallback
        if (!usingPollingRef.current && !isComplete) {
          startPolling();
        }
      };

      ws.onclose = () => {
        if (!mountedRef.current) return;
        setIsConnected(false);

        // If not complete and not already polling, start polling
        if (!isComplete && !usingPollingRef.current) {
          // Small delay before starting polling
          setTimeout(() => {
            if (mountedRef.current && !isComplete) {
              startPolling();
            }
          }, WS_RECONNECT_DELAY_MS);
        }
      };
    } catch (err) {
      console.error('Failed to create WebSocket:', err);
      // Fallback to polling
      if (!usingPollingRef.current) {
        startPolling();
      }
    }
  }, [tid, handleProgressUpdate, startPolling, isComplete]);

  /**
   * Main effect: Connect WebSocket or start polling
   */
  useEffect(() => {
    mountedRef.current = true;

    if (!tid || !enabled) {
      return;
    }

    // Reset state for new task
    setProgress({ current: 0, total: 100, percent: 0, message: '' });
    setIsComplete(false);
    setError(null);
    setStatus('pending');
    usingPollingRef.current = false;

    // Try WebSocket first
    connectWebSocket();

    // Cleanup on unmount or tid change
    return () => {
      mountedRef.current = false;

      // Close WebSocket
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }

      // Clear polling interval
      if (pollingIntervalRef.current) {
        window.clearInterval(pollingIntervalRef.current);
        pollingIntervalRef.current = null;
      }
    };
  }, [tid, enabled, connectWebSocket]);

  /**
   * Stop tracking when complete
   */
  useEffect(() => {
    if (isComplete) {
      // Close WebSocket
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }

      // Clear polling interval
      if (pollingIntervalRef.current) {
        window.clearInterval(pollingIntervalRef.current);
        pollingIntervalRef.current = null;
      }
    }
  }, [isComplete]);

  return {
    progress,
    isConnected,
    isComplete,
    error,
    status,
  };
}

export default useOptimizationProgress;
