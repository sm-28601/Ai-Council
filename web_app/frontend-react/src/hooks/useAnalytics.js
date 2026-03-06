// hooks/useAnalytics.js
import { useState, useEffect, useCallback } from 'react';
import { authAPI } from '../utils/api';

// Date range options
export const DATE_RANGES = {
  '7d': { label: 'Last 7 Days', days: 7 },
  '30d': { label: 'Last 30 Days', days: 30 },
  '90d': { label: 'Last 90 Days', days: 90 },
  'all': { label: 'All Time', days: null },
};

/**
 * Custom hook for fetching analytics data from the backend.
 * Handles loading state, error handling, data transformation, and comparison.
 *
 * @param {string} dateRange - Date range key ('7d', '30d', '90d', 'all')
 * @returns {Object} Analytics state and actions
 */
export const useAnalytics = (dateRange = '7d') => {
  const [data, setData] = useState(null);
  const [previousData, setPreviousData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchAnalytics = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      // Fetch current period data
      const params = {};
      const rangeConfig = DATE_RANGES[dateRange];
      
      if (rangeConfig?.days) {
        params.days = rangeConfig.days;
      }

      const response = await authAPI.get('/chat/analytics', { params });
      
      if (response.data.success) {
        const analytics = response.data.analytics;
        
        // Transform data for charts
        const modelsUsage = (analytics.modelsUsage || []).map((item) => ({
          name: item._id || 'Unknown',
          value: item.count,
        }));

        const executionModes = (analytics.executionModes || []).map((item) => ({
          name: item._id || 'Unknown',
          value: item.count,
        }));

        const recentActivity = (analytics.recentActivity || []).map((item) => ({
          date: item._id,
          queries: item.count,
          cost: item.totalCost,
        }));

        // Calculate cost by model (simulated from models usage and total cost)
        const totalUsage = modelsUsage.reduce((sum, m) => sum + m.value, 0);
        const costByModel = modelsUsage.map((model, index) => ({
          name: model.name,
          cost: totalUsage > 0 
            ? (model.value / totalUsage) * (analytics.totalCost || 0)
            : 0,
          queries: model.value,
        }));

        // Sort models by usage to get top models
        const topModels = [...modelsUsage].sort((a, b) => b.value - a.value).slice(0, 5);

        const currentData = {
          totalQueries: analytics.totalQueries || 0,
          totalCost: analytics.totalCost || 0,
          avgConfidence: analytics.avgConfidence || 0,
          avgExecutionTime: analytics.avgExecutionTime || 0,
          modelsUsage,
          executionModes,
          recentActivity,
          costByModel,
          topModels,
        };

        setData(currentData);

        // Fetch previous period for comparison (if not "all time")
        if (rangeConfig?.days) {
          try {
            const prevParams = {
              days: rangeConfig.days,
              offset: rangeConfig.days, // Offset by the same period
            };
            const prevResponse = await authAPI.get('/chat/analytics', { params: prevParams });
            
            if (prevResponse.data.success) {
              const prevAnalytics = prevResponse.data.analytics;
              setPreviousData({
                totalQueries: prevAnalytics.totalQueries || 0,
                totalCost: prevAnalytics.totalCost || 0,
                avgConfidence: prevAnalytics.avgConfidence || 0,
                avgExecutionTime: prevAnalytics.avgExecutionTime || 0,
              });
            }
          } catch {
            // Silently fail for comparison data - it's optional
            setPreviousData(null);
          }
        } else {
          setPreviousData(null);
        }
      } else {
        throw new Error(response.data.message || 'Failed to fetch analytics');
      }
    } catch (err) {
      const errorMessage =
        err.response?.data?.message ||
        err.message ||
        'Failed to load analytics. Please try again.';
      setError(errorMessage);
      setData(null);
      setPreviousData(null);
    } finally {
      setLoading(false);
    }
  }, [dateRange]);

  useEffect(() => {
    fetchAnalytics();
  }, [fetchAnalytics]);

  // Calculate trends (percentage change from previous period)
  const calculateTrend = useCallback((current, previous) => {
    if (!previous || previous === 0) return null;
    const change = ((current - previous) / previous) * 100;
    return {
      value: Math.abs(change).toFixed(1),
      isPositive: change >= 0,
      direction: change >= 0 ? 'up' : 'down',
    };
  }, []);

  const trends = data && previousData ? {
    totalQueries: calculateTrend(data.totalQueries, previousData.totalQueries),
    totalCost: calculateTrend(data.totalCost, previousData.totalCost),
    avgConfidence: calculateTrend(data.avgConfidence, previousData.avgConfidence),
    avgExecutionTime: calculateTrend(data.avgExecutionTime, previousData.avgExecutionTime),
  } : null;

  // Export analytics as CSV
  const exportToCSV = useCallback(() => {
    if (!data) return;

    const rows = [
      ['Metric', 'Value'],
      ['Total Queries', data.totalQueries],
      ['Total Cost ($)', data.totalCost.toFixed(4)],
      ['Average Confidence (%)', data.avgConfidence.toFixed(2)],
      ['Average Execution Time (s)', data.avgExecutionTime.toFixed(2)],
      [''],
      ['Model', 'Usage Count'],
      ...data.modelsUsage.map(m => [m.name, m.value]),
      [''],
      ['Execution Mode', 'Count'],
      ...data.executionModes.map(m => [m.name, m.value]),
      [''],
      ['Date', 'Queries', 'Cost ($)'],
      ...data.recentActivity.map(a => [a.date, a.queries, a.cost?.toFixed(4) || '0']),
    ];

    const csvContent = rows.map(row => row.join(',')).join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);
    
    link.setAttribute('href', url);
    link.setAttribute('download', `analytics_${dateRange}_${new Date().toISOString().split('T')[0]}.csv`);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }, [data, dateRange]);

  return {
    data,
    previousData,
    trends,
    loading,
    error,
    refetch: fetchAnalytics,
    exportToCSV,
  };
};

export default useAnalytics;
