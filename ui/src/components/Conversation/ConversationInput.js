import React, { useState } from 'react';
import { 
  Box, Button, Card, CardContent, TextField, Typography, 
  Paper, FormControl, InputLabel, MenuItem, Select, Grid 
} from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import DeleteIcon from '@mui/icons-material/Delete';

// Example conversations for quick testing
const SAMPLE_CONVERSATIONS = [
  {
    title: 'Internet Service Complaint',
    text: `Caller: Hello, I've been trying to get my internet fixed for a week now and nobody seems to care.
Agent: I'm sorry to hear that. Let me look into this for you.
Caller: I've called three times already and each time I was promised someone would come, but nobody ever showed up.
Agent: I apologize for the inconvenience. I can see the notes on your account.
Caller: This is ridiculous! I'm paying for a service I'm not receiving.
Agent: I understand your frustration. Let me schedule a technician visit with our highest priority.
Caller: I want a refund for the days I haven't had service.
Agent: That's a reasonable request. I'll process a credit for the days affected.`
  },
  {
    title: 'Product Upgrade (Non-complaint)',
    text: `Caller: Hi, I'm calling to upgrade my plan.
Agent: Hello! I'd be happy to help you with that. What plan are you interested in?
Caller: I saw the premium package online. It has more channels.
Agent: The premium package is a great choice. It includes 200+ channels, including HBO and Showtime.
Caller: That sounds good. How much would it cost?
Agent: The premium package is $89.99 per month. I can also offer a 3-month discount at $69.99 if you upgrade today.
Caller: That sounds like a good deal. Let's go with that.
Agent: Excellent! I'll process that upgrade right away for you.`
  },
  {
    title: 'Mixed Conversation',
    text: `Caller: Hello, I'm calling about my recent bill. It seems higher than usual.
Agent: I'd be happy to look into that for you. Can I have your account number please?
Caller: Yes, it's 123456789.
Agent: Thank you. I can see your bill is $120 this month, which is $20 more than usual.
Caller: That's right. Why is it higher?
Agent: It looks like there was a pay-per-view movie purchase on the 15th for $19.99.
Caller: I didn't order any movie! This is unacceptable.
Agent: I understand your concern. Let me check the details.
Caller: I want this charge removed immediately.
Agent: After reviewing the account, I can see this was likely an error. I'll remove the charge right away.
Caller: Thank you. I appreciate your help with this.
Agent: You're welcome. Is there anything else I can assist you with today?`
  }
];

const ConversationInput = ({ onAnalyze, processing }) => {
  const [conversation, setConversation] = useState('');
  const [chunkSize, setChunkSize] = useState(4);
  const [sampleIndex, setSampleIndex] = useState('');

  const handleAnalyze = () => {
    if (conversation.trim()) {
      onAnalyze(conversation, chunkSize);
    }
  };

  const handleClear = () => {
    setConversation('');
    setSampleIndex('');
  };

  const handleSampleChange = (event) => {
    const index = event.target.value;
    setSampleIndex(index);
    if (index !== '') {
      setConversation(SAMPLE_CONVERSATIONS[index].text);
    }
  };

  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        setConversation(e.target.result);
      };
      reader.readAsText(file);
    }
  };

  return (
    <Card variant="outlined" sx={{ mb: 4 }}>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Conversation Analysis
        </Typography>
        <Typography variant="body2" color="text.secondary" paragraph>
          Enter a conversation between a caller and agent to analyze for complaints. The conversation will be processed in chunks of {chunkSize} utterances.
        </Typography>

        <Grid container spacing={2} sx={{ mb: 2 }}>
          <Grid item xs={12} sm={6}>
            <FormControl fullWidth size="small">
              <InputLabel id="sample-conversation-label">Sample Conversation</InputLabel>
              <Select
                labelId="sample-conversation-label"
                id="sample-conversation"
                value={sampleIndex}
                label="Sample Conversation"
                onChange={handleSampleChange}
              >
                <MenuItem value="">
                  <em>Select a sample</em>
                </MenuItem>
                {SAMPLE_CONVERSATIONS.map((sample, index) => (
                  <MenuItem key={index} value={index}>
                    {sample.title}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={12} sm={6}>
            <FormControl fullWidth size="small">
              <InputLabel id="chunk-size-label">Chunk Size</InputLabel>
              <Select
                labelId="chunk-size-label"
                id="chunk-size"
                value={chunkSize}
                label="Chunk Size"
                onChange={(e) => setChunkSize(e.target.value)}
              >
                <MenuItem value={2}>2 utterances per chunk</MenuItem>
                <MenuItem value={4}>4 utterances per chunk</MenuItem>
                <MenuItem value={6}>6 utterances per chunk</MenuItem>
                <MenuItem value={8}>8 utterances per chunk</MenuItem>
              </Select>
            </FormControl>
          </Grid>
        </Grid>

        <TextField
          fullWidth
          multiline
          rows={12}
          variant="outlined"
          placeholder="Enter conversation here with format 'Speaker: Text' on each line..."
          value={conversation}
          onChange={(e) => setConversation(e.target.value)}
          sx={{ mb: 2 }}
        />

        <Paper 
          variant="outlined" 
          sx={{ 
            p: 1, 
            mb: 2, 
            display: 'flex', 
            justifyContent: 'center',
            alignItems: 'center',
            cursor: 'pointer',
            backgroundColor: 'background.default',
            transition: 'all 0.3s',
            '&:hover': {
              backgroundColor: 'action.hover',
            }
          }}
          component="label"
        >
          <input
            type="file"
            accept=".txt"
            hidden
            onChange={handleFileUpload}
          />
          <CloudUploadIcon sx={{ mr: 1 }} />
          <Typography>Upload conversation file (.txt)</Typography>
        </Paper>

        <Box sx={{ display: 'flex', gap: 2 }}>
          <Button
            variant="contained"
            color="primary"
            startIcon={<PlayArrowIcon />}
            onClick={handleAnalyze}
            disabled={!conversation.trim() || processing}
            fullWidth
          >
            Analyze Conversation
          </Button>
          <Button
            variant="outlined"
            color="secondary"
            startIcon={<DeleteIcon />}
            onClick={handleClear}
            disabled={processing}
          >
            Clear
          </Button>
        </Box>
      </CardContent>
    </Card>
  );
};

export default ConversationInput; 