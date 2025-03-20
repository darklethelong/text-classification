import React from 'react';
import { 
  Box, Card, CardContent, Typography, Chip, 
  Divider, LinearProgress, Tooltip
} from '@mui/material';
import WarningIcon from '@mui/icons-material/Warning';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';

const ConversationChunk = ({ chunk, index }) => {
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
    <Card 
      variant="outlined" 
      sx={{ 
        mb: 2,
        borderColor: isComplaint ? 'error.light' : 'success.light',
        backgroundColor: isComplaint ? 'rgba(244, 67, 54, 0.05)' : 'rgba(76, 175, 80, 0.05)'
      }}
    >
      <CardContent>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
          <Typography variant="subtitle1" component="div" sx={{ fontWeight: 'medium' }}>
            Chunk {index + 1}
          </Typography>
          <Chip
            icon={statusIcon}
            label={statusText}
            color={statusColor}
            size="small"
          />
        </Box>
        
        <Box sx={{ mb: 2 }}>
          <Tooltip title={`Confidence: ${confidencePercent}%`} arrow placement="top">
            <Box sx={{ width: '100%' }}>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 0.5 }}>
                Confidence: {confidencePercent}%
              </Typography>
              <LinearProgress
                variant="determinate"
                value={confidence * 100}
                color={isComplaint ? 'error' : 'success'}
                sx={{ height: 8, borderRadius: 4 }}
              />
            </Box>
          </Tooltip>
        </Box>

        <Divider sx={{ my: 1 }} />
        
        <Box className="chat-container" sx={{ mt: 2 }}>
          {utterances.map((utterance, i) => (
            <Box 
              key={i}
              className={`utterance ${utterance.speaker}`}
              sx={{ 
                display: 'flex',
                flexDirection: 'column',
                alignSelf: utterance.speaker === 'caller' ? 'flex-start' : 'flex-end',
              }}
            >
              <Typography 
                variant="caption" 
                sx={{ 
                  color: utterance.speaker === 'caller' ? 'primary.main' : 'success.main',
                  fontWeight: 'bold',
                  textTransform: 'capitalize'
                }}
              >
                {utterance.speaker}
              </Typography>
              <Typography variant="body2">{utterance.content}</Typography>
            </Box>
          ))}
        </Box>
      </CardContent>
    </Card>
  );
};

export default ConversationChunk; 