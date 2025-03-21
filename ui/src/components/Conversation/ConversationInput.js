import React, { useState } from 'react';
import { 
  Box, Button, Card, CardContent, TextField, Typography, 
  Paper, FormControl, InputLabel, MenuItem, Select, Grid, Divider 
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
  },
  {
    title: 'Extended Service Issues (20+ utterances)',
    text: `Caller: Hi, I'm having some serious issues with my mobile service for the past two weeks.
Agent: I'm sorry to hear that. Could you please explain what issues you're experiencing?
Caller: My calls keep dropping, and I'm getting terrible reception even in areas where I used to have full bars.
Agent: I understand how frustrating that can be. Let me check your account and the network status in your area.
Caller: Please do. It's impacting my work since I work remotely and need reliable service for client calls.
Agent: I can see there have been some network upgrades in your area recently. Can you tell me your exact location?
Caller: I'm in the downtown area, near Central Park. Zip code 10023.
Agent: Thank you. I'm checking our network status map now. It does show some maintenance was being done in that area.
Caller: That might explain it, but why wasn't I notified about this? I've been paying for service I can't use.
Agent: You should have received a text notification. Let me check if it was sent to your number.
Caller: I never received anything. This is completely unprofessional.
Agent: I apologize for that oversight. I don't see a record of the notification being sent to you, which is definitely our error.
Caller: So what are you going to do about it? I've lost business because of these dropped calls.
Agent: I completely understand your frustration. First, I'd like to credit your account for the two weeks of affected service.
Caller: That's a start, but what about the business I've lost? One of my clients actually terminated their contract.
Agent: I'm very sorry to hear that. While we can't directly compensate for lost business, I can offer you three months of our premium service package at no additional cost.
Caller: I don't know if that really helps me. What I need is reliable service.
Agent: The network upgrades should be completed by tomorrow, which should resolve your reception issues. Additionally, I can add a priority flag to your account which gives you preferred network access in congested areas.
Caller: How do I know this won't happen again next time you do "upgrades"?
Agent: Going forward, I'll personally ensure you receive advance notifications about any planned maintenance. I'm adding a note to your account right now.
Caller: I appreciate that, but I'm still not happy about losing that client.
Agent: I understand. As an additional goodwill gesture, I can waive your next month's bill entirely. Would that help?
Caller: Yes, that would be better. Thank you for working with me on this.
Agent: You're welcome. I'm processing those adjustments now. The priority network access is already active, and your account will show zero balance for next month.
Caller: When exactly will the network be back to normal?
Agent: According to our system, the work will be completed by 5 PM tomorrow. I'll also send you a follow-up text personally to confirm once it's done.
Caller: Alright, I guess that works. I still wish this hadn't happened in the first place.
Agent: I completely understand and apologize again for the inconvenience. Is there anything else I can assist you with today?
Caller: No, that's all. Just make sure the service gets fixed.
Agent: I will personally monitor this and ensure everything is resolved. Thank you for your patience, and please don't hesitate to reach out directly if you experience any further issues.`
  }
];

const textStyle = {
  color: 'var(--text-color)',
  fontSize: '1.1rem'
};

const ConversationInput = ({ onAnalyze, processing }) => {
  const [conversation, setConversation] = useState('');
  const [sampleIndex, setSampleIndex] = useState('');

  const handleAnalyze = () => {
    if (conversation.trim()) {
      onAnalyze(conversation);
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
        <Typography variant="h6" gutterBottom sx={{ ...textStyle, fontSize: '1.3rem !important' }}>
          Conversation Analysis
        </Typography>
        <Typography variant="body2" paragraph sx={{ ...textStyle }}>
          Enter a conversation between a caller and agent to analyze for complaints. The conversation will be processed in chunks of 4 utterances (required by the model).
        </Typography>

        <Grid container spacing={2} sx={{ mb: 2 }}>
          <Grid item xs={12}>
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
            hidden
            accept=".txt"
            onChange={handleFileUpload}
          />
          <CloudUploadIcon sx={{ mr: 1, color: 'var(--neon-green)' }} />
          <Typography variant="button" sx={{ color: 'var(--neon-green)' }}>
            Upload conversation file (.txt)
          </Typography>
        </Paper>

        <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
          <Button 
            variant="contained" 
            color="primary"
            startIcon={<PlayArrowIcon />}
            onClick={handleAnalyze}
            disabled={processing || !conversation.trim()}
            size="large"
            sx={{ 
              width: '80%',
              background: 'var(--neon-green)', 
              color: 'black',
              fontFamily: 'monospace',
              fontWeight: 'bold',
              '&:hover': { 
                background: 'var(--bright-green)',
              }
            }}
          >
            Analyze Conversation
          </Button>
          <Button 
            variant="outlined"
            color="error" 
            onClick={handleClear}
            disabled={processing || !conversation.trim()}
            sx={{
              color: 'var(--neon-green)',
              borderColor: 'var(--neon-green)',
              '&:hover': {
                borderColor: 'var(--neon-green)',
                backgroundColor: 'rgba(0, 255, 65, 0.1)'
              }
            }}
          >
            <DeleteIcon />
          </Button>
        </Box>
        
        {/* Cyberpunk Info Panel to fill empty space */}
        <Paper
          sx={{
            mt: 3,
            p: 2,
            backgroundColor: 'rgba(0, 20, 0, 0.7)',
            border: '1px solid var(--neon-green)',
            boxShadow: '0 0 10px rgba(0, 255, 65, 0.3)',
            color: 'var(--neon-green)',
            borderRadius: '4px',
            fontFamily: 'monospace',
          }}
        >
          <Typography sx={{ fontSize: '1rem', fontWeight: 'bold', mb: 1, color: 'var(--neon-green)', textShadow: '0 0 5px rgba(0, 255, 65, 0.5)' }}>
            // SYSTEM GUIDE //
          </Typography>
          
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
            <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 1 }}>
              <Typography component="span" sx={{ color: 'var(--bright-green)' }}>{'>'}</Typography>
              <Typography variant="body2" sx={{ color: 'var(--text-color)' }}>
                Format each line as <span style={{ color: 'var(--bright-green)' }}>Caller: </span> or <span style={{ color: 'var(--bright-green)' }}>Agent: </span> followed by the text
              </Typography>
            </Box>
            
            <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 1 }}>
              <Typography component="span" sx={{ color: 'var(--bright-green)' }}>{'>'}</Typography>
              <Typography variant="body2" sx={{ color: 'var(--text-color)' }}>
                Analysis works best with conversations containing 8+ utterances
              </Typography>
            </Box>
            
            <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 1 }}>
              <Typography component="span" sx={{ color: 'var(--bright-green)' }}>{'>'}</Typography>
              <Typography variant="body2" sx={{ color: 'var(--text-color)' }}>
                The model analyzes conversations in chunks of 4 utterances to detect complaints
              </Typography>
            </Box>
            
            <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 1 }}>
              <Typography component="span" sx={{ color: 'var(--bright-green)' }}>{'>'}</Typography>
              <Typography variant="body2" sx={{ color: 'var(--text-color)' }}>
                For best results, include complete caller-agent interactions
              </Typography>
            </Box>
          </Box>
          
          <Divider sx={{ my: 2, backgroundColor: 'rgba(0, 255, 65, 0.3)' }} />
          
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <Typography variant="caption" sx={{ color: 'var(--text-color)', opacity: 0.7 }}>
              Complaint detection v3.5.2
            </Typography>
            <Typography variant="caption" sx={{ color: 'var(--neon-green)', opacity: 0.7 }}>
              [AUTHORIZED ACCESS ONLY]
            </Typography>
          </Box>
        </Paper>
        
      </CardContent>
    </Card>
  );
};

export default ConversationInput; 