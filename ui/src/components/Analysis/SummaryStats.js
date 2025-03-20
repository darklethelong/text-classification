import React from 'react';
import { 
  Box, Card, CardContent, Typography, Grid, 
  Divider, LinearProgress, Paper
} from '@mui/material';
import WarningAmberIcon from '@mui/icons-material/WarningAmber';
import ChatIcon from '@mui/icons-material/Chat';
import PersonIcon from '@mui/icons-material/Person';
import ReportProblemIcon from '@mui/icons-material/ReportProblem';

const StatCard = ({ title, value, icon, color, subtitle }) => {
  return (
    <Paper elevation={1} sx={{ p: 2, height: '100%' }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
        <Typography variant="subtitle2" color="text.secondary">
          {title}
        </Typography>
        <Box sx={{ 
          backgroundColor: `${color}.light`, 
          color: `${color}.dark`,
          p: 0.5,
          borderRadius: '50%',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center'
        }}>
          {icon}
        </Box>
      </Box>
      <Typography variant="h4" component="div" sx={{ fontWeight: 'bold' }}>
        {value}
      </Typography>
      {subtitle && (
        <Typography variant="body2" color="text.secondary">
          {subtitle}
        </Typography>
      )}
    </Paper>
  );
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
  
  return (
    <Card variant="outlined" sx={{ mb: 4 }}>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Analysis Summary
        </Typography>
        
        <Typography variant="body2" color="text.secondary" paragraph>
          Overview of complaint analysis for the conversation.
        </Typography>
        
        <Box sx={{ mb: 3 }}>
          <Typography variant="subtitle2" gutterBottom>
            Overall Complaint Level: {complaintPercentage.toFixed(1)}%
          </Typography>
          <LinearProgress
            variant="determinate"
            value={complaintPercentage}
            color={severityColor}
            sx={{ height: 10, borderRadius: 5 }}
          />
          <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 0.5 }}>
            <Typography variant="caption" color="text.secondary">
              0% (No complaints)
            </Typography>
            <Typography variant="caption" color="text.secondary">
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
      </CardContent>
    </Card>
  );
};

export default SummaryStats; 