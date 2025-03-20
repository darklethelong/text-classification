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
  
  return (
    <AppBar position="sticky" elevation={1} color="default">
      <Container maxWidth="xl">
        <Toolbar disableGutters>
          <AnalyticsIcon sx={{ mr: 1, color: 'primary.main' }} />
          <Typography
            variant="h6"
            noWrap
            sx={{
              flexGrow: 1,
              fontWeight: 700,
              color: 'primary.main',
              textDecoration: 'none',
            }}
          >
            Complaint Detection UI
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