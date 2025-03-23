import React from 'react';
import { 
  Box, Card, CardContent, Typography, Divider, Grid, 
  LinearProgress, Paper, Chip
} from '@mui/material';
import WarningAmberIcon from '@mui/icons-material/WarningAmber';
import ReportProblemIcon from '@mui/icons-material/ReportProblem';
import ChatIcon from '@mui/icons-material/Chat';
import PersonIcon from '@mui/icons-material/Person';
import AccessTimeIcon from '@mui/icons-material/AccessTime';

// Component for displaying a statistic card
const StatCard = ({ title, value, icon, color, subtitle }) => (
  <Paper
    elevation={1}
    sx={{
      p: 2,
      display: 'flex',
      flexDirection: 'column',
      height: '100%',
      borderLeft: 3,
      borderColor: `${color}.main`,
    }}
  >
    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
      <Typography variant="subtitle2" component="div" gutterBottom sx={{ fontWeight: 'medium' }}>
        {title}
      </Typography>
      <Box sx={{ color: `${color}.main` }}>{icon}</Box>
    </Box>
    <Typography variant="h5" component="div" sx={{ fontWeight: 'bold', my: 1, color: `${color}.main` }}>
      {value}
    </Typography>
    <Typography variant="caption" color="text.secondary">
      {subtitle}
    </Typography>
  </Paper>
);

// Add custom text style
const textStyle = {
  color: 'var(--text-color)',
  fontSize: '1.1rem'
};

const SummaryStats = ({ analyzedData }) => {
  const { chunks, complaintPercentage, utterances } = analyzedData;
  
  // Additional statistics
  const totalUtterances = utterances.length;
  const callerUtterances = utterances.filter(u => u.speaker === 'caller').length;
  const agentUtterances = utterances.filter(u => u.speaker === 'agent').length;
  
  const complaintChunks = chunks.filter(chunk => chunk.isComplaint).length;
  const totalChunks = chunks.length;
  
  // Complaint utterances (just for display - we actually analyze at chunk level)
  const callerComplaintUtterances = Math.round(callerUtterances * (complaintPercentage / 100));
  
  // Severity calculation based on complaint percentage
  let severity = 'Low';
  let severityColor = 'success';
  
  if (complaintPercentage >= 70) {
    severity = 'High';
    severityColor = 'error';
  } else if (complaintPercentage >= 30) {
    severity = 'Medium';
    severityColor = 'warning';
  }

  // Get timestamp information
  let earliestTimestamp = null;
  let latestTimestamp = null;
  
  if (chunks && chunks.length > 0) {
    // Find chunks with timestamps
    const chunksWithTimestamps = chunks.filter(chunk => chunk.timestamp);
    
    if (chunksWithTimestamps.length > 0) {
      // Find earliest and latest timestamps
      earliestTimestamp = new Date(Math.min(...chunksWithTimestamps.map(c => new Date(c.timestamp))));
      latestTimestamp = new Date(Math.max(...chunksWithTimestamps.map(c => new Date(c.timestamp))));
    }
  }
  
  return (
    <Card variant="outlined" sx={{ mb: 4 }}>
      <CardContent>
        <Typography variant="h6" gutterBottom sx={{ ...textStyle, fontSize: '1.3rem !important' }}>
          Analysis Summary
        </Typography>
        
        <Typography variant="body2" paragraph sx={{ ...textStyle }}>
          Overview of complaint analysis for the conversation.
        </Typography>
        
        <Box sx={{ mb: 3 }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
            <Typography variant="subtitle2" sx={{ ...textStyle }}>
              Overall Complaint Level: {complaintPercentage.toFixed(1)}%
            </Typography>
            <Chip 
              label={severity}
              color={severityColor}
              size="small"
              icon={<WarningAmberIcon />}
              sx={{ color: 'var(--text-color)', '& .MuiChip-label': { fontSize: '1.1rem' } }}
            />
          </Box>
          <LinearProgress
            variant="determinate"
            value={complaintPercentage}
            color={severityColor}
            sx={{ height: 10, borderRadius: 5 }}
          />
          <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 0.5 }}>
            <Typography variant="caption" sx={{ ...textStyle }}>
              0% (No complaints)
            </Typography>
            <Typography variant="caption" sx={{ ...textStyle }}>
              100% (All complaints)
            </Typography>
          </Box>
        </Box>
        
        <Divider sx={{ my: 2 }} />
        
        <Grid container spacing={2}>
          <Grid item xs={12} sm={6} md={3}>
            <StatCard
              title="Complaint Severity"
              value={severity}
              icon={<WarningAmberIcon />}
              color={severityColor}
              subtitle="Based on percentage"
            />
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <StatCard
              title="Complaint Chunks"
              value={`${complaintChunks}/${totalChunks}`}
              icon={<ReportProblemIcon />}
              color="error"
              subtitle={`${(complaintChunks / totalChunks * 100).toFixed(1)}% of chunks`}
            />
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <StatCard
              title="Total Utterances"
              value={totalUtterances}
              icon={<ChatIcon />}
              color="info"
              subtitle={`${callerUtterances} caller, ${agentUtterances} agent`}
            />
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <StatCard
              title="Caller Complaints"
              value={`~${callerComplaintUtterances}/${callerUtterances}`}
              icon={<PersonIcon />}
              color="warning"
              subtitle="Estimated from chunks"
            />
          </Grid>
        </Grid>
        
        {earliestTimestamp && latestTimestamp && (
          <>
            <Divider sx={{ my: 2 }} />
            
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 2 }}>
              <AccessTimeIcon color="action" />
              <Typography 
                variant="body1" 
                sx={{ 
                  ...textStyle,
                  fontFamily: 'monospace',
                  fontWeight: 'medium' 
                }}
              >
                Analysis period: {earliestTimestamp.toLocaleTimeString()} - {latestTimestamp.toLocaleTimeString()} 
                ({((latestTimestamp - earliestTimestamp) / 1000).toFixed(1)} seconds)
              </Typography>
            </Box>
          </>
        )}
      </CardContent>
    </Card>
  );
};

export default SummaryStats; 