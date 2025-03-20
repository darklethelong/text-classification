import axios from 'axios';

// Create axios instance
const api = axios.create({
  baseURL: process.env.REACT_APP_API_URL || 'http://localhost:8000',
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add interceptor to include auth token in requests
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Auth services
const authService = {
  login: async (username, password) => {
    try {
      const formData = new URLSearchParams();
      formData.append('username', username);
      formData.append('password', password);
      
      const response = await api.post('/token', formData, {
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        },
      });
      
      const { access_token, token_type } = response.data;
      localStorage.setItem('token', access_token);
      return { success: true, token: access_token };
    } catch (error) {
      console.error('Login error:', error);
      return { 
        success: false, 
        error: error.response?.data?.detail || 'Authentication failed'
      };
    }
  },
  
  logout: () => {
    localStorage.removeItem('token');
    return { success: true };
  },
  
  isAuthenticated: () => {
    return localStorage.getItem('token') !== null;
  },
  
  getUser: async () => {
    try {
      const response = await api.get('/users/me');
      return response.data;
    } catch (error) {
      console.error('Get user error:', error);
      return null;
    }
  },
};

// Prediction services
const predictionService = {
  predict: async (text) => {
    try {
      const response = await api.post('/analyze', { text });
      return {
        success: true,
        data: response.data,
      };
    } catch (error) {
      console.error('Prediction error:', error);
      return {
        success: false,
        error: error.response?.data?.detail || 'Failed to process prediction',
      };
    }
  },
  
  checkHealth: async () => {
    try {
      const response = await api.get('/');
      return {
        success: true,
        data: response.data,
      };
    } catch (error) {
      console.error('Health check error:', error);
      return {
        success: false,
        error: 'API service is unavailable',
      };
    }
  },
};

export { api, authService, predictionService }; 