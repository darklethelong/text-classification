import React, { useState } from 'react';
import { 
  Container, Box, Typography, Alert, CircularProgress,
  Divider, Accordion, AccordionSummary, AccordionDetails
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';

import Header from '../Layout/Header';
import ConversationInput from '../Conversation/ConversationInput';
import ConversationChunk from '../Conversation/ConversationChunk';
import SummaryStats from '../Analysis/SummaryStats';
import ComplaintChart from '../Analysis/ComplaintChart';

import { 
  parseConversation, 
  chunkConversation, 
  calculateComplaintPercentage 
} from '../../utils/conversationParser';
import { predictionService, authService } from '../../services/api';

const Dashboard = ({ onLogout }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [analyzedData, setAnalyzedData] = useState(null);
  const [expandedChunk, setExpandedChunk] = useState('');
  
  const handleExpandChange = (chunkId) => (event, isExpanded) => {
    setExpandedChunk(isExpanded ? chunkId : '');
  };
  
  // Load user data
  React.useEffect(() => {
    const fetchUser = async () => {
      const userData = await authService.getUser();
      setUser(userData);
    };
    
    fetchUser();
  }, []);
  
  const handleAnalyzeConversation = async (conversationText, chunkSize) => {
    setLoading(true);
    setError('');
    
    try {
      // Parse conversation into utterances
      const utterances = parseConversation(conversationText);
      
      // Create chunks of utterances for analysis
      const chunks = chunkConversation(utterances, chunkSize);
      
      // Process each chunk with the API
      const processedChunks = [];
      
      for (const chunk of chunks) {
        // Call API for prediction
        const result = await predictionService.predict(chunk.text);
        
        if (result.success) {
          // Update chunk with prediction result
          const updatedChunk = {
            ...chunk,
            isComplaint: result.data.is_complaint,
            confidence: result.data.confidence,
          };
          
          processedChunks.push(updatedChunk);
        } else {
          throw new Error(`Failed to process chunk: ${result.error}`);
        }
      }
      
      // Calculate overall complaint percentage
      const complaintPercentage = calculateComplaintPercentage(processedChunks);
      
      // Set analyzed data
      setAnalyzedData({
        utterances,
        chunks: processedChunks,
        complaintPercentage,
      });
      
      // Expand the first chunk by default
      if (processedChunks.length > 0) {
        setExpandedChunk(processedChunks[0].id);
      }
      
    } catch (err) {
      console.error('Analysis error:', err);
      setError(`Failed to analyze conversation: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };
  
  return (
    <Box sx={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      <Header user={user} onLogout={onLogout} />
      
      <Container maxWidth="lg" sx={{ py: 4, flex: 1 }}>
        <ConversationInput onAnalyze={handleAnalyzeConversation} processing={loading} />
        
        {loading && (
          <Box sx={{ display: 'flex', justifyContent: 'center', my: 4 }}>
            <CircularProgress />
          </Box>
        )}
        
        {error && (
          <Alert severity="error" sx={{ mb: 4 }}>
            {error}
          </Alert>
        )}
        
        {analyzedData && !loading && (
          <Box>
            <SummaryStats analyzedData={analyzedData} />
            <ComplaintChart analyzedData={analyzedData} />
            
            <Typography variant="h6" gutterBottom sx={{ mb: 2 }}>
              Conversation Analysis Details
            </Typography>
            
            <Typography variant="body2" color="text.secondary" paragraph>
              Click on each chunk to view the detailed conversation and analysis.
            </Typography>
            
            <Divider sx={{ mb: 2 }} />
            
            {analyzedData.chunks.map((chunk, index) => (
              <Accordion 
                key={chunk.id} 
                expanded={expandedChunk === chunk.id}
                onChange={handleExpandChange(chunk.id)}
                sx={{ mb: 1 }}
              >
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Typography 
                    sx={{ 
                      fontWeight: chunk.isComplaint ? 'bold' : 'normal',
                      color: chunk.isComplaint ? 'error.main' : 'text.primary'
                    }}
                  >
                    Chunk {index + 1} {chunk.isComplaint ? '(Complaint Detected)' : '(No Complaint)'}
                  </Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <ConversationChunk chunk={chunk} index={index} />
                </AccordionDetails>
              </Accordion>
            ))}
          </Box>
        )}
      </Container>
      
      <Box 
        component="footer" 
        sx={{ 
          p: 2, 
          mt: 'auto', 
          backgroundColor: 'background.paper',
          borderTop: '1px solid',
          borderColor: 'divider'
        }}
      >
        <Typography variant="body2" color="text.secondary" align="center">
          Complaint Detection API Testing UI | © {new Date().getFullYear()}
        </Typography>
      </Box>
    </Box>
  );
};

export default Dashboard; 