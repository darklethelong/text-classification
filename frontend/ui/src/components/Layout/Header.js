import React from 'react';
import { 
  AppBar, Box, Toolbar, Typography, Button, 
  useTheme, useMediaQuery, Container, Chip
} from '@mui/material';
import AnalyticsIcon from '@mui/icons-material/Analytics';
import LogoutIcon from '@mui/icons-material/Logout';
import PersonIcon from '@mui/icons-material/Person';

const Header = ({ user, onLogout }) => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));
  
  const textStyle = {
    color: 'var(--text-color)',
    fontSize: '1.2rem'
  };

  return (
    <AppBar position="sticky" elevation={1} color="default">
      <Container maxWidth="xl">
        <Toolbar disableGutters>
          <AnalyticsIcon sx={{ mr: 1, color: 'primary.main' }} />
          <Typography 
            variant="h6" 
            component="div" 
            sx={{ 
              ...textStyle,
              flexGrow: 1,
              fontFamily: 'monospace',
              fontWeight: 'bold',
              textShadow: '0 0 10px var(--neon-green)',
              letterSpacing: '.1rem',
              fontSize: '1.5rem !important'
            }}
          >
            COMPLAINT DETECTION UI
          </Typography>
          
          {user && (
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
              {!isMobile && (
                <Chip
                  icon={<PersonIcon />}
                  label={`${user.username}`}
                  color="primary"
                  variant="outlined"
                  size="small"
                />
              )}
              <Button 
                variant="text" 
                color="inherit"
                onClick={onLogout}
                startIcon={<LogoutIcon />}
                size={isMobile ? 'small' : 'medium'}
              >
                {isMobile ? '' : 'Logout'}
              </Button>
            </Box>
          )}
        </Toolbar>
      </Container>
    </AppBar>
  );
};

export default Header; 