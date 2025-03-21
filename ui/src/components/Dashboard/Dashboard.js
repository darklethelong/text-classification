import React, { useState, useEffect, useRef } from 'react';
import { 
  Container, Box, Typography, Alert, CircularProgress,
  Divider, Accordion, AccordionSummary, AccordionDetails,
  Button, Grid, Paper, useTheme
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import TimelineIcon from '@mui/icons-material/Timeline';
import DashboardIcon from '@mui/icons-material/Dashboard';
import CodeIcon from '@mui/icons-material/Code';
import TerminalIcon from '@mui/icons-material/Terminal';
import ChatIcon from '@mui/icons-material/Chat';

import Header from '../Layout/Header';
import ConversationInput from '../Conversation/ConversationInput';
import ConversationChunk from '../Conversation/ConversationChunk';
import SummaryStats from '../Analysis/SummaryStats';
import ComplaintTrackingChart from '../Analysis/ComplaintTrackingChart';
import MatrixRain from '../Effects/MatrixRain';

import { 
  parseConversation, 
  chunkConversation, 
  calculateComplaintPercentage,
  reconstructFromApiResponse
} from '../../utils/conversationParser';
import { predictionService, authService } from '../../services/api';

// Create a cyberpunk styled section container
const CyberpunkSection = ({ title, icon, children }) => {
  return (
    <Paper
      elevation={0}
      sx={{
        position: 'relative',
        borderRadius: 2,
        overflow: 'hidden',
        backgroundColor: 'rgba(10, 2, 5, 0.95)',
        backdropFilter: 'none',
        color: 'var(--neon-green)',
        border: '2px solid var(--neon-green)',
        boxShadow: '0 0 15px rgba(0, 255, 65, 0.5)',
        height: '100%',
        display: 'flex',
        flexDirection: 'column'
      }}
    >
      <Box 
        sx={{ 
          display: 'flex', 
          alignItems: 'center', 
          p: 1.5, 
          borderBottom: '1px solid var(--dark-green)',
          background: 'linear-gradient(90deg, rgba(0,20,0,0.8) 0%, rgba(0,40,0,0.4) 100%)',
        }}
      >
        <Box sx={{ 
          mr: 1, 
          color: 'var(--neon-green)', 
          display: 'flex' 
        }}>
          {icon}
        </Box>
        <Typography 
          variant="h6" 
          sx={{ 
            fontFamily: 'monospace', 
            textTransform: 'uppercase',
            letterSpacing: '0.1em',
            fontSize: '1.2rem',
            fontWeight: 'bold',
            textShadow: '0 0 5px var(--neon-green)',
            color: 'var(--text-color)'
          }}
        >
          {title}
        </Typography>
      </Box>
      <Box sx={{ 
        flex: 1, 
        p: 2,
      }}>
        {children}
      </Box>
    </Paper>
  );
};

// Create a real-time tracker component (floating)
const RealTimeTracker = ({ chunks, onClose }) => {
  const chartRef = useRef(null);
  const [chartData, setChartData] = useState([]);
  
  useEffect(() => {
    if (chunks && chunks.length > 0) {
      // Convert chunks to time-series data for the chart
      const timeSeriesData = chunks.map((chunk, index) => ({
        timestamp: new Date(chunk.timestamp || Date.now()),
        confidence: chunk.confidence * 100,
        isComplaint: chunk.isComplaint,
        chunkId: chunk.id || `chunk-${index}`
      }));
      
      // Sort by timestamp
      const sortedData = timeSeriesData.sort((a, b) => a.timestamp - b.timestamp);
      setChartData(sortedData);
    }
  }, [chunks]);
  
  return (
    <Paper 
      elevation={3} 
      sx={{ 
        position: 'fixed', 
        bottom: 20, 
        right: 20, 
        width: 350, 
        height: 300, 
        p: 0,
        zIndex: 1000,
        display: 'flex',
        flexDirection: 'column',
        border: '1px solid var(--neon-green)',
        boxShadow: '0 0 20px rgba(0, 255, 65, 0.4)',
        background: 'rgba(13, 2, 8, 0.9)',
        backdropFilter: 'none',
        overflow: 'hidden'
      }}
    >
      <Box sx={{ 
        display: 'flex', 
        justifyContent: 'space-between', 
        alignItems: 'center', 
        p: 1.5,
        borderBottom: '1px solid var(--dark-green)',
        background: 'linear-gradient(90deg, rgba(0,20,0,0.8) 0%, rgba(0,40,0,0.4) 100%)',
      }}>
        <Typography 
          variant="h6" 
          sx={{ 
            display: 'flex', 
            alignItems: 'center',
            fontFamily: 'monospace',
            fontSize: '0.9rem',
            textTransform: 'uppercase'
          }}
        >
          <TimelineIcon sx={{ mr: 1, color: 'var(--neon-green)' }} /> 
          Real-time Tracking
        </Typography>
        <Button 
          variant="text" 
          size="small" 
          onClick={onClose}
          sx={{
            color: 'var(--neon-green)',
            '&:hover': {
              backgroundColor: 'rgba(0, 255, 65, 0.1)'
            }
          }}
        >
          Close
        </Button>
      </Box>
      
      <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'auto', p: 2 }}>
        {chartData.length > 0 ? (
          <Box sx={{ height: 200 }} ref={chartRef}>
            {chartData.map((point, index) => (
              <Box key={index} sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                <Typography variant="caption" sx={{ color: 'var(--muted-green)' }}>
                  {point.timestamp.toLocaleTimeString()}
                </Typography>
                <Typography 
                  variant="caption" 
                  sx={{ 
                    color: point.isComplaint ? 'var(--bright-green)' : 'var(--dark-green)',
                    fontWeight: 'bold',
                    textShadow: point.isComplaint ? '0 0 5px var(--neon-green)' : 'none'
                  }}
                >
                  {point.confidence.toFixed(1)}% {point.isComplaint ? '(Complaint)' : '(No Complaint)'}
                </Typography>
              </Box>
            ))}
          </Box>
        ) : (
          <Typography variant="body2" sx={{ textAlign: 'center', mt: 4, color: 'var(--muted-green)' }}>
            No data available yet. Analyze a conversation to see real-time tracking.
          </Typography>
        )}
      </Box>
    </Paper>
  );
};

const Dashboard = ({ onLogout }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [analyzedData, setAnalyzedData] = useState(null);
  const [expandedChunk, setExpandedChunk] = useState('');
  const [showTracker, setShowTracker] = useState(false);
  const [showMatrixEffect, setShowMatrixEffect] = useState(true);
  const [matrixMode, setMatrixMode] = useState('matrix'); // 'matrix', 'binary', 'custom'
  
  const handleExpandChange = (chunkId) => (event, isExpanded) => {
    setExpandedChunk(isExpanded ? chunkId : '');
  };
  
  // Cycle through matrix modes
  const cycleMatrixMode = () => {
    setMatrixMode(prev => {
      switch(prev) {
        case 'matrix': return 'binary';
        case 'binary': return 'custom';
        default: return 'matrix';
      }
    });
  };
  
  // Load user data
  useEffect(() => {
    const fetchUser = async () => {
      const userData = await authService.getUser();
      setUser(userData);
    };
    
    fetchUser();
  }, []);
  
  const handleAnalyzeConversation = async (conversationText) => {
    setLoading(true);
    setError('');
    
    try {
      // Log the input text to debug
      console.log('Analyzing conversation text:', conversationText);
      console.log('Conversation utterance count:', conversationText.split('\n').filter(line => line.trim()).length);
      
      // Use the API endpoint to analyze the entire conversation with sliding window approach
      // Fixed chunk size to 4 as required by the model
      const result = await predictionService.analyzeConversation(conversationText);
      
      console.log('API Response:', result);
      
      if (result.success) {
        // Log the raw API response
        console.log('Raw API chunks received:', result.data.chunks);
        console.log('Chunk count:', result.data.chunks.length);
        
        // Import the reconstructFromApiResponse if needed
        // Use our utility to process the sliding window data properly
        const processedData = reconstructFromApiResponse(result.data);
        
        console.log('Processed data:', processedData);
        
        // Set the analyzed data with the processed results
        setAnalyzedData({
          ...processedData,
          complaintPercentage: result.data.complaint_percentage
        });
        
        // Show the tracker by default
        setShowTracker(true);
        
        // Expand the first chunk by default if we have chunks
        if (processedData.chunks.length > 0) {
          setExpandedChunk(processedData.chunks[0].id);
        }
      } else {
        throw new Error(result.error);
      }
    } catch (err) {
      console.error('Analysis error:', err);
      setError(`Failed to analyze conversation: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };
  
  return (
    <Box sx={{ 
      minHeight: '100vh',
      display: 'flex', 
      flexDirection: 'column',
      background: 'linear-gradient(135deg, rgba(13, 2, 8, 0.90) 0%, rgba(18, 18, 18, 0.90) 100%)',
      backgroundSize: 'cover',
      backgroundAttachment: 'fixed',
      overflowX: 'hidden',
      position: 'relative',
      zIndex: 1
    }}>
      {showMatrixEffect && (
        <MatrixRain 
          opacity={0.2} 
          speed={1.2} 
          density={0.98} 
          fontSize={14}
          characterSet={matrixMode}
        />
      )}
      <Header user={user} onLogout={onLogout} />
      
      <Container maxWidth="xl" sx={{ py: 3, flex: 1 }}>
        {/* Top control bar for global actions */}
        {!loading && analyzedData && (
          <Box sx={{ mb: 2, display: 'flex', gap: 2 }}>
            <Button 
              variant="contained"
              color="primary"
              onClick={() => setShowTracker(!showTracker)}
              startIcon={<TimelineIcon />}
              sx={{
                fontFamily: 'monospace',
                textTransform: 'uppercase',
                letterSpacing: '0.05em'
              }}
            >
              {showTracker ? "HIDE_TRACKER" : "SHOW_TRACKER"}
            </Button>
            
            <Button 
              variant="contained"
              color="secondary"
              onClick={() => setShowMatrixEffect(!showMatrixEffect)}
              startIcon={<CodeIcon />}
              sx={{
                fontFamily: 'monospace',
                textTransform: 'uppercase',
                letterSpacing: '0.05em'
              }}
            >
              {showMatrixEffect ? "DISABLE_MATRIX" : "ENABLE_MATRIX"}
            </Button>
            
            {showMatrixEffect && (
              <Button 
                variant="contained"
                color="info"
                onClick={cycleMatrixMode}
                startIcon={<TerminalIcon />}
                sx={{
                  fontFamily: 'monospace',
                  textTransform: 'uppercase',
                  letterSpacing: '0.05em'
                }}
              >
                MATRIX_MODE: {matrixMode.toUpperCase()}
              </Button>
            )}
          </Box>
        )}
        
        {/* Loading indicator */}
        {loading && (
          <Box sx={{ 
            display: 'flex', 
            justifyContent: 'center', 
            alignItems: 'center',
            height: '80vh'
          }}>
            <CircularProgress 
              sx={{ 
                color: 'var(--neon-green)',
                '& .MuiCircularProgress-circle': {
                  strokeWidth: 3
                },
                boxShadow: '0 0 20px var(--neon-green)'
              }} 
              size={60} 
            />
            <Typography 
              variant="h6" 
              sx={{ 
                ml: 2, 
                fontFamily: 'monospace',
                textTransform: 'uppercase',
                animation: 'pulse 1.5s infinite',
                fontSize: '1.2rem',
                color: 'var(--text-color)',
                '@keyframes pulse': {
                  '0%': {
                    opacity: 0.6,
                  },
                  '50%': {
                    opacity: 1,
                  },
                  '100%': {
                    opacity: 0.6,
                  }
                }
              }}
            >
              Processing...
            </Typography>
          </Box>
        )}
        
        {/* Error message */}
        {error && (
          <Alert 
            severity="error" 
            sx={{ 
              mb: 4,
              backgroundColor: 'rgba(211, 47, 47, 0.1)', 
              color: '#ff5252',
              border: '1px solid #ff5252',
              '& .MuiAlert-icon': {
                color: '#ff5252'
              }
            }}
          >
            {error}
          </Alert>
        )}
        
        {/* Centered Conversation Input when no analysis is present */}
        {!loading && !analyzedData && (
          <Box sx={{ 
            display: 'flex', 
            flexDirection: 'column',
            justifyContent: 'center', 
            alignItems: 'center',
            minHeight: '80vh'
          }}>
            <Box sx={{ 
              display: 'flex', 
              justifyContent: 'center', 
              mb: 3,
              gap: 2
            }}>
              <Button 
                variant="contained"
                color="secondary"
                onClick={() => setShowMatrixEffect(!showMatrixEffect)}
                startIcon={<CodeIcon />}
                sx={{
                  fontFamily: 'monospace',
                  textTransform: 'uppercase',
                  letterSpacing: '0.05em'
                }}
              >
                {showMatrixEffect ? "DISABLE_MATRIX" : "ENABLE_MATRIX"}
              </Button>
              
              {showMatrixEffect && (
                <Button 
                  variant="contained"
                  color="info"
                  onClick={cycleMatrixMode}
                  startIcon={<TerminalIcon />}
                  sx={{
                    fontFamily: 'monospace',
                    textTransform: 'uppercase',
                    letterSpacing: '0.05em'
                  }}
                >
                  MATRIX_MODE: {matrixMode.toUpperCase()}
                </Button>
              )}
            </Box>
            
            <Box sx={{ width: '100%', maxWidth: 800 }}>
              <CyberpunkSection title="Conversation Analysis" icon={<CodeIcon />}>
                <ConversationInput onAnalyze={handleAnalyzeConversation} processing={loading} />
              </CyberpunkSection>
            </Box>
          </Box>
        )}
        
        {/* Grid Layout after analysis */}
        {!loading && analyzedData && (
          <Grid container spacing={3}>
            {/* Top Left: Conversation Analysis (Input) */}
            <Grid item xs={12} md={6}>
              <CyberpunkSection title="Conversation Analysis" icon={<CodeIcon />}>
                <ConversationInput onAnalyze={handleAnalyzeConversation} processing={loading} />
              </CyberpunkSection>
            </Grid>
            
            {/* Top Right: TERMINAL // CUSTOMER COMPLAINT MATRIX (Chart) */}
            <Grid item xs={12} md={6}>
              <CyberpunkSection title="Terminal // Customer Complaint Matrix" icon={<TerminalIcon />}>
                {analyzedData && <ComplaintTrackingChart analyzedData={analyzedData} />}
              </CyberpunkSection>
            </Grid>
            
            {/* Bottom Left: Analysis Summary */}
            <Grid item xs={12} md={6}>
              <CyberpunkSection title="Analysis Summary" icon={<DashboardIcon />}>
                {analyzedData && <SummaryStats analyzedData={analyzedData} />}
              </CyberpunkSection>
            </Grid>
            
            {/* Bottom Right: Conversation Analysis Details */}
            <Grid item xs={12} md={6}>
              <CyberpunkSection title="Conversation Analysis Details" icon={<ChatIcon />}>
                {analyzedData && (
                  <>
                    <Typography 
                      variant="body2" 
                      sx={{ 
                        mb: 2, 
                        color: 'text.secondary',
                        fontFamily: 'monospace',
                        fontSize: '0.85rem',
                      }}
                    >
                      // Expand chunks to view conversation details
                    </Typography>
                    {analyzedData.chunks.length > 0 ? (
                      <Box>
                        {analyzedData.chunks.map((chunk, index) => (
                          <ConversationChunk 
                            key={chunk.id}
                            chunk={chunk}
                            index={index}
                            expanded={expandedChunk === chunk.id}
                            onChange={handleExpandChange(chunk.id)}
                          />
                        ))}
                      </Box>
                    ) : (
                      <Typography variant="body1" sx={{ textAlign: 'center', mt: 2, color: 'text.secondary' }}>
                        No valid conversation chunks detected.
                      </Typography>
                    )}
                  </>
                )}
              </CyberpunkSection>
            </Grid>
          </Grid>
        )}
        
        {/* Complaint Tracker */}
        {showTracker && analyzedData && (
          <RealTimeTracker 
            chunks={analyzedData.chunks}
            onClose={() => setShowTracker(false)}
          />
        )}
      </Container>
    </Box>
  );
};

export default Dashboard; 