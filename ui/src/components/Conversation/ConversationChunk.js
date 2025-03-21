import React from 'react';
import { 
  Box, Card, CardContent, Typography, Chip, 
  Divider, LinearProgress, Tooltip, Accordion, AccordionSummary, AccordionDetails
} from '@mui/material';
import WarningIcon from '@mui/icons-material/Warning';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';

const ConversationChunk = ({ chunk, index, expanded, onChange }) => {
  const { isComplaint, confidence, utterances } = chunk;
  
  // Format confidence as percentage
  const confidencePercent = (confidence * 100).toFixed(1);
  
  // Determine color based on complaint status and confidence
  let statusColor, statusIcon, statusText;
  
  if (isComplaint) {
    statusColor = 'error';
    statusIcon = <WarningIcon fontSize="small" />;
    statusText = 'Complaint Detected';
  } else {
    statusColor = 'success';
    statusIcon = <CheckCircleIcon fontSize="small" />;
    statusText = 'No Complaint';
  }
  
  return (
    <Accordion 
      expanded={expanded}
      onChange={onChange}
      sx={{ 
        mb: 1,
        background: 'rgba(13, 2, 8, 0.95)',
        backdropFilter: 'none',
        border: `2px solid ${isComplaint ? '#39ff14' : 'var(--neon-green)'}`,
        boxShadow: isComplaint ? '0 0 10px rgba(0, 255, 65, 0.5)' : 'none',
        transition: 'all 0.3s ease',
        '&:hover': {
          boxShadow: '0 0 15px rgba(0, 255, 65, 0.6)'
        }
      }}
    >
      <AccordionSummary 
        expandIcon={<ExpandMoreIcon sx={{ color: 'var(--neon-green)', fontSize: '1.5rem' }} />}
        sx={{
          borderBottom: expanded ? '2px solid var(--neon-green)' : 'none',
          background: isComplaint 
            ? 'linear-gradient(90deg, rgba(0,40,10,0.8) 0%, rgba(0,60,0,0.4) 100%)'
            : 'linear-gradient(90deg, rgba(0,20,0,0.8) 0%, rgba(0,40,0,0.4) 100%)',
          '&:hover': {
            backgroundColor: isComplaint ? 'rgba(0,40,10,0.9)' : 'rgba(0,20,0,0.9)',
          },
          transition: 'all 0.3s ease'
        }}
      >
        <Typography 
          sx={{ 
            fontFamily: 'monospace',
            fontWeight: 'bold',
            color: isComplaint ? '#39ff14' : 'var(--neon-green)',
            textShadow: isComplaint ? '0 0 10px var(--neon-green)' : '0 0 5px var(--neon-green)',
            fontSize: '1.2rem',
            letterSpacing: '0.05em'
          }}
        >
          CHUNK_{index + 1} {isComplaint ? '[COMPLAINT_DETECTED]' : '[STATUS_NORMAL]'} // 
          {new Date(chunk.timestamp).toLocaleTimeString()}
        </Typography>
      </AccordionSummary>
      <AccordionDetails
        sx={{
          backgroundColor: '#000000',
          p: 0,
          border: '2px solid var(--neon-green)',
          borderTop: 'none'
        }}
      >
        <Card 
          variant="outlined" 
          sx={{ 
            border: 'none',
            boxShadow: 'none',
            backgroundColor: '#000000'
          }}
        >
          <CardContent sx={{ p: 2, backgroundColor: '#000000' }}>
            <Box sx={{ mb: 3 }}>
              <Tooltip title={`Confidence: ${confidencePercent}%`} arrow placement="top">
                <Box sx={{ width: '100%' }}>
                  <Typography 
                    variant="body2" 
                    sx={{ 
                      mb: 1, 
                      color: '#FFFFFF', 
                      fontSize: '1.3rem',
                      fontWeight: 'bold',
                      letterSpacing: '0.03em'
                    }}
                  >
                    Confidence: {confidencePercent}%
                  </Typography>
                  <LinearProgress
                    variant="determinate"
                    value={confidence * 100}
                    sx={{ 
                      height: 15, 
                      borderRadius: 10,
                      backgroundColor: '#111111',
                      border: '2px solid #00ff65',
                      boxShadow: '0 0 20px #00ff65',
                      '& .MuiLinearProgress-bar': {
                        backgroundColor: '#00ff65',
                        boxShadow: '0 0 15px #00ff65'
                      }
                    }}
                  />
                </Box>
              </Tooltip>
            </Box>

            <Divider sx={{ my: 2, backgroundColor: '#00ff65', opacity: 0.8, height: '2px' }} />
            
            <Typography 
              variant="subtitle1" 
              sx={{ 
                color: '#FFFFFF', 
                fontFamily: 'monospace', 
                mb: 2, 
                fontWeight: 'bold',
                fontSize: '1.4rem',
                letterSpacing: '0.08em',
                textAlign: 'center',
                backgroundColor: '#002200',
                p: 1.5,
                borderRadius: 1,
                border: '2px solid #00ff65'
              }}
            >
              CONVERSATION TRANSCRIPT
            </Typography>
            
            <Box 
              className="chat-container" 
              sx={{ 
                mt: 2, 
                p: 1.5, 
                backgroundColor: '#000000',
                borderRadius: 2,
                border: '2px solid #00ff65'
              }}
            >
              {utterances.map((utterance, i) => (
                <Box 
                  key={i}
                  className={`utterance ${utterance.speaker}`}
                  sx={{ 
                    display: 'flex',
                    flexDirection: 'column',
                    mb: 2,
                    p: 2.5,
                    borderRadius: 2,
                    maxWidth: '80%',
                    backgroundColor: utterance.speaker === 'caller' 
                      ? '#004400' 
                      : '#003366',
                    alignSelf: utterance.speaker === 'caller' ? 'flex-start' : 'flex-end',
                    border: utterance.speaker === 'caller'
                      ? '2px solid #00ff65'
                      : '2px solid #00ccff',
                    boxShadow: utterance.speaker === 'caller'
                      ? '0 0 15px #00ff65'
                      : '0 0 15px #00ccff',
                  }}
                >
                  <Typography 
                    variant="caption" 
                    sx={{ 
                      color: utterance.speaker === 'caller' ? '#5FFF5F' : '#5FFFFF',
                      fontWeight: 'bold',
                      textTransform: 'uppercase',
                      fontFamily: 'monospace',
                      fontSize: '1.3rem',
                      letterSpacing: '0.08em',
                      mb: 0.8
                    }}
                  >
                    {utterance.speaker}
                  </Typography>
                  <Typography 
                    variant="body2" 
                    sx={{ 
                      color: '#FFFFFF', 
                      wordBreak: 'break-word',
                      fontSize: '1.2rem',
                      fontWeight: 700,
                      lineHeight: 1.4,
                      letterSpacing: '0.02em'
                    }}
                  >
                    {utterance.content}
                  </Typography>
                </Box>
              ))}
            </Box>
          </CardContent>
        </Card>
      </AccordionDetails>
    </Accordion>
  );
};

export default ConversationChunk; 