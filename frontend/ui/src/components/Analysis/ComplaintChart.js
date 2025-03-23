import React, { useEffect, useState } from 'react';
import { 
  Card, CardContent, Typography, Box,
  ToggleButtonGroup, ToggleButton
} from '@mui/material';
import { 
  Chart as ChartJS, 
  CategoryScale, 
  LinearScale, 
  PointElement, 
  LineElement, 
  BarElement,
  Title, 
  Tooltip, 
  Legend, 
  Filler
} from 'chart.js';
import { Line, Bar } from 'react-chartjs-2';
import CompareArrowsIcon from '@mui/icons-material/CompareArrows';
import StackedBarChartIcon from '@mui/icons-material/StackedBarChart';
import TimelineIcon from '@mui/icons-material/Timeline';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

const ComplaintChart = ({ analyzedData }) => {
  const [chartType, setChartType] = useState('line');
  const [chartData, setChartData] = useState(null);
  
  const handleChartTypeChange = (event, newChartType) => {
    if (newChartType !== null) {
      setChartType(newChartType);
    }
  };
  
  useEffect(() => {
    if (!analyzedData || !analyzedData.chunks || analyzedData.chunks.length === 0) {
      return;
    }
    
    const { chunks } = analyzedData;
    
    // Sort chunks by timestamp if available
    const sortedChunks = [...chunks].sort((a, b) => {
      const timeA = a.timestamp ? new Date(a.timestamp) : 0;
      const timeB = b.timestamp ? new Date(b.timestamp) : 0;
      return timeA - timeB;
    });
    
    // Prepare data for chart
    const labels = sortedChunks.map((chunk, index) => {
      if (chunk.timestamp) {
        const date = new Date(chunk.timestamp);
        return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
      }
      return `Chunk ${index + 1}`;
    });
    
    const confidenceData = sortedChunks.map(chunk => chunk.confidence * 100);
    const isComplaintData = sortedChunks.map(chunk => chunk.isComplaint ? 100 : 0);
    
    // Create gradient for area chart
    const getGradient = (ctx, chartArea) => {
      if (!ctx || !chartArea) return null;
      const gradient = ctx.createLinearGradient(0, chartArea.bottom, 0, chartArea.top);
      gradient.addColorStop(0, 'rgba(63, 81, 181, 0.0)');
      gradient.addColorStop(1, 'rgba(63, 81, 181, 0.3)');
      return gradient;
    };
    
    // Format for stock-like line chart
    const lineChartData = {
      labels,
      datasets: [
        {
          label: 'Complaint Confidence (%)',
          data: confidenceData,
          borderColor: 'rgba(63, 81, 181, 1)',
          backgroundColor: function(context) {
            const chart = context.chart;
            const {ctx, chartArea} = chart;
            if (!chartArea) return null;
            return getGradient(ctx, chartArea);
          },
          tension: 0.4,
          fill: true,
          pointBackgroundColor: sortedChunks.map(chunk => 
            chunk.isComplaint ? 'rgba(244, 67, 54, 1)' : 'rgba(76, 175, 80, 1)'
          ),
          pointRadius: 5,
          pointHoverRadius: 7,
          borderWidth: 2,
        }
      ]
    };
    
    // Format for bar chart
    const barChartData = {
      labels,
      datasets: [
        {
          label: 'Complaint Confidence (%)',
          data: confidenceData,
          backgroundColor: sortedChunks.map(chunk => 
            chunk.isComplaint ? 'rgba(244, 67, 54, 0.7)' : 'rgba(76, 175, 80, 0.7)'
          ),
          borderColor: sortedChunks.map(chunk => 
            chunk.isComplaint ? 'rgba(244, 67, 54, 1)' : 'rgba(76, 175, 80, 1)'
          ),
          borderWidth: 1,
        }
      ]
    };
    
    // Format for comparison chart
    const comparisonChartData = {
      labels,
      datasets: [
        {
          label: 'Complaint Status',
          data: isComplaintData,
          backgroundColor: 'rgba(244, 67, 54, 0.7)',
          borderColor: 'rgba(244, 67, 54, 1)',
          borderWidth: 1,
          stack: 'Stack 0',
        },
        {
          label: 'Confidence (%)',
          data: confidenceData,
          backgroundColor: 'rgba(63, 81, 181, 0.7)',
          borderColor: 'rgba(63, 81, 181, 1)',
          borderWidth: 1,
          stack: 'Stack 1',
        }
      ]
    };
    
    // Format for stock-like area chart
    const stockChartData = {
      labels,
      datasets: [
        {
          label: 'Complaint Confidence (%)',
          data: confidenceData,
          borderColor: 'rgba(33, 150, 243, 1)',
          backgroundColor: function(context) {
            const chart = context.chart;
            const {ctx, chartArea} = chart;
            if (!chartArea) return null;
            const gradient = ctx.createLinearGradient(0, chartArea.bottom, 0, chartArea.top);
            gradient.addColorStop(0, 'rgba(33, 150, 243, 0.0)');
            gradient.addColorStop(1, 'rgba(33, 150, 243, 0.3)');
            return gradient;
          },
          tension: 0.3,
          fill: true,
          pointBackgroundColor: sortedChunks.map(chunk => 
            chunk.isComplaint ? 'rgba(244, 67, 54, 1)' : 'rgba(76, 175, 80, 1)'
          ),
          pointRadius: 4,
          pointHoverRadius: 6,
          borderWidth: 2,
        }
      ]
    };
    
    // Set chart data based on type
    if (chartType === 'line') {
      setChartData(lineChartData);
    } else if (chartType === 'bar') {
      setChartData(barChartData);
    } else if (chartType === 'comparison') {
      setChartData(comparisonChartData);
    } else if (chartType === 'stock') {
      setChartData(stockChartData);
    }
    
  }, [analyzedData, chartType]);
  
  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      y: {
        beginAtZero: true,
        max: 100,
        grid: {
          color: 'rgba(0, 0, 0, 0.05)'
        },
        title: {
          display: true,
          text: 'Percentage (%)'
        }
      },
      x: {
        grid: {
          display: false
        },
        title: {
          display: true,
          text: 'Timestamp'
        }
      }
    },
    plugins: {
      legend: {
        position: 'top',
      },
      tooltip: {
        callbacks: {
          label: function(context) {
            let label = context.dataset.label || '';
            if (label) {
              label += ': ';
            }
            if (context.parsed.y !== null) {
              label += `${context.parsed.y.toFixed(1)}%`;
            }
            return label;
          },
          footer: function(tooltipItems) {
            if (chartType === 'line' || chartType === 'bar' || chartType === 'stock') {
              const index = tooltipItems[0].dataIndex;
              const isComplaint = analyzedData.chunks[index].isComplaint;
              return `Status: ${isComplaint ? 'Complaint' : 'Non-complaint'}`;
            }
            return '';
          }
        }
      }
    },
    animation: {
      duration: 1000
    }
  };
  
  if (!chartData) {
    return (
      <Card variant="outlined" sx={{ mb: 4 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Complaint Trend Analysis
          </Typography>
          <Typography variant="body2" color="text.secondary">
            No data available for charting. Please analyze a conversation first.
          </Typography>
        </CardContent>
      </Card>
    );
  }
  
  return (
    <Card variant="outlined" sx={{ mb: 4 }}>
      <CardContent>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h6">
            Complaint Trend Analysis
          </Typography>
          <ToggleButtonGroup
            value={chartType}
            exclusive
            onChange={handleChartTypeChange}
            size="small"
          >
            <ToggleButton value="line" aria-label="line chart">
              <TimelineIcon fontSize="small" />
            </ToggleButton>
            <ToggleButton value="stock" aria-label="stock chart">
              <TrendingUpIcon fontSize="small" />
            </ToggleButton>
            <ToggleButton value="bar" aria-label="bar chart">
              <StackedBarChartIcon fontSize="small" />
            </ToggleButton>
            <ToggleButton value="comparison" aria-label="comparison chart">
              <CompareArrowsIcon fontSize="small" />
            </ToggleButton>
          </ToggleButtonGroup>
        </Box>
        
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          {chartType === 'line' 
            ? 'Line chart showing complaint confidence across conversation chunks. Red points indicate detected complaints.'
            : chartType === 'bar'
            ? 'Bar chart showing complaint confidence levels for each chunk. Red bars indicate detected complaints.'
            : chartType === 'stock'
            ? 'Stock-like chart showing complaint confidence trend over time. Red points indicate detected complaints.'
            : 'Comparison chart showing both complaint status and confidence levels.'}
        </Typography>
        
        <Box sx={{ height: 300 }}>
          {(chartType === 'line' || chartType === 'stock') && <Line data={chartData} options={chartOptions} />}
          {chartType === 'bar' && <Bar data={chartData} options={chartOptions} />}
          {chartType === 'comparison' && <Bar data={chartData} options={chartOptions} />}
        </Box>
      </CardContent>
    </Card>
  );
};

export default ComplaintChart; 