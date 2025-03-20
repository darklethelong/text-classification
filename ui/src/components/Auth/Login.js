import React, { useState, useEffect } from 'react';
import { 
  Box, Button, Card, CardContent, TextField, Typography, 
  CircularProgress, Alert, Stack 
} from '@mui/material';
import LockOutlinedIcon from '@mui/icons-material/LockOutlined';
import { authService } from '../../services/api';

const Login = ({ onLoginSuccess }) => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [checkingAuth, setCheckingAuth] = useState(true);

  // Check if already authenticated
  useEffect(() => {
    const checkAuth = async () => {
      if (authService.isAuthenticated()) {
        try {
          const user = await authService.getUser();
          if (user) {
            onLoginSuccess();
          }
        } catch (error) {
          console.error('Error checking authentication:', error);
          // If token is invalid, clear it
          authService.logout();
        }
      }
      setCheckingAuth(false);
    };

    checkAuth();
  }, [onLoginSuccess]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      const result = await authService.login(username, password);
      if (result.success) {
        onLoginSuccess();
      } else {
        setError(result.error);
      }
    } catch (err) {
      setError('Authentication failed. Please try again.');
      console.error('Login error:', err);
    } finally {
      setLoading(false);
    }
  };

  if (checkingAuth) {
    return (
      <Box 
        sx={{ 
          display: 'flex', 
          justifyContent: 'center', 
          alignItems: 'center',
          height: '100vh' 
        }}
      >
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        minHeight: '100vh',
        padding: 2,
        backgroundColor: 'background.default',
      }}
    >
      <Card sx={{ maxWidth: 400, width: '100%' }}>
        <CardContent>
          <Box
            sx={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              mb: 2,
            }}
          >
            <Box
              sx={{
                bgcolor: 'primary.main',
                color: 'white',
                borderRadius: '50%',
                p: 1,
                mb: 2,
              }}
            >
              <LockOutlinedIcon fontSize="large" />
            </Box>
            <Typography component="h1" variant="h5">
              Sign in to Complaint Detection UI
            </Typography>
          </Box>

          {error && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {error}
            </Alert>
          )}

          <Box component="form" onSubmit={handleSubmit} noValidate>
            <TextField
              margin="normal"
              required
              fullWidth
              id="username"
              label="Username"
              name="username"
              autoComplete="username"
              autoFocus
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              disabled={loading}
            />
            <TextField
              margin="normal"
              required
              fullWidth
              name="password"
              label="Password"
              type="password"
              id="password"
              autoComplete="current-password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              disabled={loading}
            />
            <Button
              type="submit"
              fullWidth
              variant="contained"
              sx={{ mt: 3, mb: 2 }}
              disabled={loading || !username || !password}
            >
              {loading ? <CircularProgress size={24} /> : 'Sign In'}
            </Button>
            <Stack direction="row" justifyContent="center">
              <Typography variant="body2" color="text.secondary">
                Default credentials: admin / adminpassword
              </Typography>
            </Stack>
          </Box>
        </CardContent>
      </Card>
    </Box>
  );
};

export default Login; 