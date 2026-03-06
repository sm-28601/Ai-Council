import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  ArrowLeft,
  TrendingUp,
  TrendingDown,
  DollarSign,
  Zap,
  Brain,
  RefreshCw,
  Download,
  Calendar,
  ChevronDown,
  Award,
  Info,
} from 'lucide-react';
import {
  PieChart,
  Pie,
  Cell,
  BarChart,
  Bar,
  LineChart,
  Line,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { LoadingSpinner } from '../components/LoadingSpinner';
import { ErrorMessage } from '../components/Errormessage';
import { SkeletonCard } from '../components/SkeletonCard';
import { useAnalytics, DATE_RANGES } from '../hooks/useAnalytics';

// Chart color palettes
const COLORS = ['#3B82F6', '#10B981', '#8B5CF6', '#F59E0B', '#EF4444', '#06B6D4'];
const GRADIENT_COLORS = {
  primary: '#3B82F6',
  secondary: '#10B981',
};

// Animated counter component
const AnimatedNumber = ({ value, prefix = '', suffix = '', decimals = 0, duration = 1000 }) => {
  const [displayValue, setDisplayValue] = useState(0);
  const startTime = useRef(null);
  const animationRef = useRef(null);

  useEffect(() => {
    const numValue = typeof value === 'number' ? value : parseFloat(value) || 0;
    
    const animate = (timestamp) => {
      if (!startTime.current) startTime.current = timestamp;
      const progress = Math.min((timestamp - startTime.current) / duration, 1);
      
      // Easing function for smooth animation
      const easeOutQuart = 1 - Math.pow(1 - progress, 4);
      setDisplayValue(numValue * easeOutQuart);

      if (progress < 1) {
        animationRef.current = requestAnimationFrame(animate);
      }
    };

    startTime.current = null;
    animationRef.current = requestAnimationFrame(animate);

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [value, duration]);

  return (
    <span>
      {prefix}
      {displayValue.toFixed(decimals)}
      {suffix}
    </span>
  );
};

// Trend indicator component
const TrendIndicator = ({ trend, invertColors = false }) => {
  if (!trend) return null;

  const isPositiveGood = !invertColors;
  const isGood = isPositiveGood ? trend.isPositive : !trend.isPositive;
  const colorClass = isGood ? 'text-green-600 bg-green-50' : 'text-red-600 bg-red-50';
  const Icon = trend.direction === 'up' ? TrendingUp : TrendingDown;

  return (
    <div className={`inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium ${colorClass}`}>
      <Icon className="w-3 h-3" />
      <span>{trend.value}%</span>
    </div>
  );
};

// Tooltip wrapper component
const InfoTooltip = ({ text }) => {
  const [show, setShow] = useState(false);

  return (
    <div className="relative inline-block">
      <Info
        className="w-4 h-4 text-gray-400 cursor-help"
        onMouseEnter={() => setShow(true)}
        onMouseLeave={() => setShow(false)}
      />
      {show && (
        <div className="absolute z-10 bottom-full left-1/2 transform -translate-x-1/2 mb-2 px-3 py-2 bg-gray-900 text-white text-xs rounded-lg whitespace-nowrap">
          {text}
          <div className="absolute top-full left-1/2 transform -translate-x-1/2 border-4 border-transparent border-t-gray-900" />
        </div>
      )}
    </div>
  );
};

const Analytics = () => {
  const navigate = useNavigate();
  const [dateRange, setDateRange] = useState('7d');
  const [showDateDropdown, setShowDateDropdown] = useState(false);
  
  const { data: analyticsData, trends, loading, error, refetch, exportToCSV } = useAnalytics(dateRange);

  const hasData = analyticsData && analyticsData.totalQueries > 0;

  const stats = [
    {
      label: 'Total Queries',
      value: analyticsData?.totalQueries ?? 0,
      displayValue: analyticsData?.totalQueries ?? 0,
      icon: Brain,
      color: 'text-blue-600',
      bgColor: 'bg-blue-50',
      trend: trends?.totalQueries,
      tooltip: 'Total number of queries made in this period',
      decimals: 0,
    },
    {
      label: 'Total Cost',
      value: analyticsData?.totalCost ?? 0,
      displayValue: analyticsData?.totalCost ?? 0,
      prefix: '$',
      icon: DollarSign,
      color: 'text-green-600',
      bgColor: 'bg-green-50',
      trend: trends?.totalCost,
      invertTrend: true, // Lower cost is better
      tooltip: 'Total API cost for all queries',
      decimals: 4,
    },
    {
      label: 'Avg Confidence',
      value: analyticsData?.avgConfidence ?? 0,
      displayValue: analyticsData?.avgConfidence ?? 0,
      suffix: '%',
      icon: TrendingUp,
      color: 'text-purple-600',
      bgColor: 'bg-purple-50',
      trend: trends?.avgConfidence,
      tooltip: 'Average confidence score of AI responses',
      decimals: 1,
    },
    {
      label: 'Avg Time',
      value: analyticsData?.avgExecutionTime ?? 0,
      displayValue: analyticsData?.avgExecutionTime ?? 0,
      suffix: 's',
      icon: Zap,
      color: 'text-orange-600',
      bgColor: 'bg-orange-50',
      trend: trends?.avgExecutionTime,
      invertTrend: true, // Lower time is better
      tooltip: 'Average response time per query',
      decimals: 2,
    },
  ];

  // Custom tooltip for charts
  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-white p-3 rounded-lg shadow-lg border border-gray-200">
          <p className="font-semibold text-gray-800 mb-1">{label}</p>
          {payload.map((entry, index) => (
            <p key={index} style={{ color: entry.color }} className="text-sm">
              {entry.name}: {typeof entry.value === 'number' && entry.name.includes('Cost') 
                ? `$${entry.value.toFixed(4)}` 
                : entry.value}
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  // Custom label for pie charts
  const renderCustomLabel = ({ name, percent }) => {
    if (percent < 0.05) return null; // Don't show label for small slices
    return `${name} (${(percent * 100).toFixed(0)}%)`;
  };

  return (
    <div className="min-h-screen p-6 bg-gray-50">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="bg-white rounded-2xl shadow-lg p-4 mb-6">
          <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
            <div className="flex items-center gap-4">
              <button
                onClick={() => navigate('/dashboard')}
                className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
              >
                <ArrowLeft className="w-6 h-6" />
              </button>
              <h1 className="text-2xl font-bold text-gray-800">Analytics Dashboard</h1>
            </div>
            
            <div className="flex items-center gap-3 w-full sm:w-auto">
              {/* Date Range Filter */}
              <div className="relative flex-1 sm:flex-none">
                <button
                  onClick={() => setShowDateDropdown(!showDateDropdown)}
                  className="flex items-center gap-2 px-4 py-2 bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors w-full sm:w-auto justify-between"
                >
                  <Calendar className="w-4 h-4 text-gray-600" />
                  <span className="text-sm font-medium text-gray-700">
                    {DATE_RANGES[dateRange].label}
                  </span>
                  <ChevronDown className={`w-4 h-4 text-gray-600 transition-transform ${showDateDropdown ? 'rotate-180' : ''}`} />
                </button>
                
                {showDateDropdown && (
                  <div className="absolute top-full left-0 right-0 sm:right-auto mt-2 bg-white rounded-lg shadow-lg border border-gray-200 py-1 z-20 min-w-[160px]">
                    {Object.entries(DATE_RANGES).map(([key, { label }]) => (
                      <button
                        key={key}
                        onClick={() => {
                          setDateRange(key);
                          setShowDateDropdown(false);
                        }}
                        className={`w-full text-left px-4 py-2 text-sm hover:bg-gray-100 transition-colors ${
                          dateRange === key ? 'bg-primary-50 text-primary-700 font-medium' : 'text-gray-700'
                        }`}
                      >
                        {label}
                      </button>
                    ))}
                  </div>
                )}
              </div>

              {/* Export Button */}
              <button
                onClick={exportToCSV}
                disabled={loading || !hasData}
                className="flex items-center gap-2 px-4 py-2 bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                title="Export to CSV"
              >
                <Download className="w-4 h-4 text-gray-600" />
                <span className="hidden sm:inline text-sm font-medium text-gray-700">Export</span>
              </button>

              {/* Refresh Button */}
              <button
                onClick={refetch}
                disabled={loading}
                className="flex items-center gap-2 px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors disabled:opacity-50"
              >
                <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
                <span className="hidden sm:inline">Refresh</span>
              </button>
            </div>
          </div>
        </div>

        {/* Error Message */}
        {error && (
          <div className="mb-6">
            <ErrorMessage error={error} onRetry={refetch} />
          </div>
        )}

        {/* Stats Grid */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 md:gap-6 mb-6">
          {loading
            ? [...Array(4)].map((_, i) => <SkeletonCard key={i} />)
            : stats.map((stat, index) => (
                <div key={index} className="bg-white rounded-2xl shadow-lg p-6 hover:shadow-xl transition-shadow">
                  <div className="flex items-center justify-between mb-4">
                    <div className={`p-3 rounded-xl ${stat.bgColor}`}>
                      <stat.icon className={`w-6 h-6 ${stat.color}`} />
                    </div>
                    <div className="flex items-center gap-2">
                      {stat.trend && <TrendIndicator trend={stat.trend} invertColors={stat.invertTrend} />}
                      <InfoTooltip text={stat.tooltip} />
                    </div>
                  </div>
                  <p className="text-2xl md:text-3xl font-bold text-gray-800 mb-1">
                    <AnimatedNumber
                      value={stat.displayValue}
                      prefix={stat.prefix || ''}
                      suffix={stat.suffix || ''}
                      decimals={stat.decimals}
                    />
                  </p>
                  <p className="text-gray-600 text-sm">{stat.label}</p>
                </div>
              ))}
        </div>

        {/* Main Content Area */}
        {loading ? (
          <div className="bg-white rounded-2xl shadow-lg p-12">
            <LoadingSpinner size="lg" text="Loading your analytics..." />
          </div>
        ) : !hasData ? (
          <div className="bg-white rounded-2xl shadow-lg p-12 text-center">
            <Brain className="w-16 h-16 text-gray-300 mx-auto mb-4" />
            <h3 className="text-xl font-bold text-gray-800 mb-2">No Data Yet</h3>
            <p className="text-gray-600 mb-6">
              Start chatting with AI Council to see your analytics here!
            </p>
            <button
              onClick={() => navigate('/chat')}
              className="px-6 py-3 bg-gradient-to-r from-primary-600 to-primary-700 text-white rounded-lg font-semibold hover:from-primary-700 hover:to-primary-800 transition-all"
            >
              Start Chatting
            </button>
          </div>
        ) : (
          <div className="space-y-6">
            {/* Top Models & Cost Breakdown Row */}
            <div className="grid lg:grid-cols-2 gap-6">
              {/* Top Models */}
              {analyticsData.topModels?.length > 0 && (
                <div className="bg-white rounded-2xl shadow-lg p-6">
                  <div className="flex items-center gap-2 mb-4">
                    <Award className="w-5 h-5 text-yellow-500" />
                    <h3 className="text-lg font-bold text-gray-800">Most Used Models</h3>
                  </div>
                  <div className="space-y-3">
                    {analyticsData.topModels.map((model, index) => (
                      <div key={model.name} className="flex items-center gap-3">
                        <div className={`w-8 h-8 rounded-full flex items-center justify-center text-white font-bold text-sm ${
                          index === 0 ? 'bg-yellow-500' : index === 1 ? 'bg-gray-400' : index === 2 ? 'bg-amber-600' : 'bg-gray-300'
                        }`}>
                          {index + 1}
                        </div>
                        <div className="flex-1">
                          <div className="flex justify-between items-center mb-1">
                            <span className="font-medium text-gray-800 text-sm truncate">{model.name}</span>
                            <span className="text-sm text-gray-600">{model.value} queries</span>
                          </div>
                          <div className="h-2 bg-gray-100 rounded-full overflow-hidden">
                            <div
                              className="h-full rounded-full transition-all duration-500"
                              style={{
                                width: `${(model.value / analyticsData.topModels[0].value) * 100}%`,
                                backgroundColor: COLORS[index % COLORS.length],
                              }}
                            />
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Cost Breakdown by Model */}
              {analyticsData.costByModel?.length > 0 && (
                <div className="bg-white rounded-2xl shadow-lg p-6">
                  <div className="flex items-center gap-2 mb-4">
                    <DollarSign className="w-5 h-5 text-green-500" />
                    <h3 className="text-lg font-bold text-gray-800">Cost Breakdown by Model</h3>
                  </div>
                  <ResponsiveContainer width="100%" height={250}>
                    <PieChart>
                      <Pie
                        data={analyticsData.costByModel}
                        cx="50%"
                        cy="50%"
                        innerRadius={60}
                        outerRadius={90}
                        fill="#8884d8"
                        dataKey="cost"
                        nameKey="name"
                      >
                        {analyticsData.costByModel.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                        ))}
                      </Pie>
                      <Tooltip
                        formatter={(value) => [`$${Number(value).toFixed(4)}`, 'Cost']}
                      />
                      <Legend />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
              )}
            </div>

            {/* Charts Row 1: Recent Activity */}
            {analyticsData.recentActivity?.length > 0 && (
              <div className="grid lg:grid-cols-2 gap-6">
                {/* Line Chart - Queries Over Time */}
                <div className="bg-white rounded-2xl shadow-lg p-6">
                  <h3 className="text-lg font-bold text-gray-800 mb-4">Queries Over Time</h3>
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={analyticsData.recentActivity}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
                      <XAxis
                        dataKey="date"
                        tick={{ fontSize: 12 }}
                        tickFormatter={(value) => new Date(value).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
                      />
                      <YAxis tick={{ fontSize: 12 }} />
                      <Tooltip content={<CustomTooltip />} />
                      <Legend />
                      <Line
                        type="monotone"
                        dataKey="queries"
                        name="Queries"
                        stroke={GRADIENT_COLORS.primary}
                        strokeWidth={3}
                        dot={{ fill: GRADIENT_COLORS.primary, strokeWidth: 2 }}
                        activeDot={{ r: 6 }}
                        animationDuration={1000}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>

                {/* Area Chart - Cost Over Time */}
                <div className="bg-white rounded-2xl shadow-lg p-6">
                  <h3 className="text-lg font-bold text-gray-800 mb-4">Cost Over Time</h3>
                  <ResponsiveContainer width="100%" height={300}>
                    <AreaChart data={analyticsData.recentActivity}>
                      <defs>
                        <linearGradient id="colorCost" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor={GRADIENT_COLORS.secondary} stopOpacity={0.8} />
                          <stop offset="95%" stopColor={GRADIENT_COLORS.secondary} stopOpacity={0.1} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
                      <XAxis
                        dataKey="date"
                        tick={{ fontSize: 12 }}
                        tickFormatter={(value) => new Date(value).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
                      />
                      <YAxis tick={{ fontSize: 12 }} tickFormatter={(value) => `$${value.toFixed(2)}`} />
                      <Tooltip content={<CustomTooltip />} />
                      <Legend />
                      <Area
                        type="monotone"
                        dataKey="cost"
                        name="Cost ($)"
                        stroke={GRADIENT_COLORS.secondary}
                        fillOpacity={1}
                        fill="url(#colorCost)"
                        animationDuration={1000}
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}

            {/* Charts Row 2: Distribution Charts */}
            <div className="grid lg:grid-cols-2 gap-6">
              {/* Pie Chart - Models Usage */}
              {analyticsData.modelsUsage?.length > 0 && (
                <div className="bg-white rounded-2xl shadow-lg p-6">
                  <h3 className="text-lg font-bold text-gray-800 mb-4">Model Usage Distribution</h3>
                  <ResponsiveContainer width="100%" height={300}>
                    <PieChart>
                      <Pie
                        data={analyticsData.modelsUsage}
                        cx="50%"
                        cy="50%"
                        labelLine={false}
                        label={renderCustomLabel}
                        outerRadius={100}
                        fill="#8884d8"
                        dataKey="value"
                        animationDuration={1000}
                      >
                        {analyticsData.modelsUsage.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                        ))}
                      </Pie>
                      <Tooltip />
                      <Legend />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
              )}

              {/* Bar Chart - Execution Modes */}
              {analyticsData.executionModes?.length > 0 && (
                <div className="bg-white rounded-2xl shadow-lg p-6">
                  <h3 className="text-lg font-bold text-gray-800 mb-4">Execution Mode Distribution</h3>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={analyticsData.executionModes} layout="vertical">
                      <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
                      <XAxis type="number" tick={{ fontSize: 12 }} />
                      <YAxis dataKey="name" type="category" width={100} tick={{ fontSize: 12 }} />
                      <Tooltip content={<CustomTooltip />} />
                      <Legend />
                      <Bar
                        dataKey="value"
                        name="Queries"
                        fill={GRADIENT_COLORS.primary}
                        radius={[0, 4, 4, 0]}
                        animationDuration={1000}
                      >
                        {analyticsData.executionModes.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              )}
            </div>

            {/* Models Usage Bar Chart (alternative view) */}
            {analyticsData.modelsUsage?.length > 0 && (
              <div className="bg-white rounded-2xl shadow-lg p-6">
                <h3 className="text-lg font-bold text-gray-800 mb-4">Model Usage Comparison</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={analyticsData.modelsUsage}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
                    <XAxis dataKey="name" tick={{ fontSize: 12 }} />
                    <YAxis tick={{ fontSize: 12 }} />
                    <Tooltip content={<CustomTooltip />} />
                    <Legend />
                    <Bar
                      dataKey="value"
                      name="Usage Count"
                      fill={GRADIENT_COLORS.primary}
                      radius={[4, 4, 0, 0]}
                      animationDuration={1000}
                    >
                      {analyticsData.modelsUsage.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default Analytics;
