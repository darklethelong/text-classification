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
    
    // Prepare data for chart
    const labels = chunks.map((_, index) => `Chunk ${index + 1}`);
    const confidenceData = chunks.map(chunk => chunk.confidence * 100);
    const isComplaintData = chunks.map(chunk => chunk.isComplaint ? 100 : 0);
    
    // Format for line chart
    const lineChartData = {
      labels,
      datasets: [
        {
          label: 'Complaint Confidence (%)',
          data: confidenceData,
          borderColor: 'rgba(63, 81, 181, 1)',
          backgroundColor: 'rgba(63, 81, 181, 0.2)',
          tension: 0.3,
          fill: true,
          pointBackgroundColor: chunks.map(chunk => 
            chunk.isComplaint ? 'rgba(244, 67, 54, 1)' : 'rgba(76, 175, 80, 1)'
          ),
          pointRadius: 5,
          pointHoverRadius: 7,
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
          backgroundColor: chunks.map(chunk => 
            chunk.isComplaint ? 'rgba(244, 67, 54, 0.7)' : 'rgba(76, 175, 80, 0.7)'
          ),
          borderColor: chunks.map(chunk => 
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
    
    // Set chart data based on type
    if (chartType === 'line') {
      setChartData(lineChartData);
    } else if (chartType === 'bar') {
      setChartData(barChartData);
    } else if (chartType === 'comparison') {
      setChartData(comparisonChartData);
    }
    
  }, [analyzedData, chartType]);
  
  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      y: {
        beginAtZero: true,
        max: 100,
        title: {
          display: true,
          text: 'Percentage (%)'
        }
      },
      x: {
        title: {
          display: true,
          text: 'Conversation Chunks'
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
            if (chartType === 'line' || chartType === 'bar') {
              const index = tooltipItems[0].dataIndex;
              const isComplaint = analyzedData.chunks[index].isComplaint;
              return `Status: ${isComplaint ? 'Complaint' : 'Non-complaint'}`;
            }
            return '';
          }
        }
      }
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
            : 'Comparison chart showing both complaint status and confidence levels.'}
        </Typography>
        
        <Box sx={{ height: 300 }}>
          {chartType === 'line' && <Line data={chartData} options={chartOptions} />}
          {chartType === 'bar' && <Bar data={chartData} options={chartOptions} />}
          {chartType === 'comparison' && <Bar data={chartData} options={chartOptions} />}
        </Box>
      </CardContent>
    </Card>
  );
};

export default ComplaintChart; 