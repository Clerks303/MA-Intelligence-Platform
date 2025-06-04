import axios from 'axios';
import tokenStorage from '../utils/tokenStorage';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api/v1';

console.log('🌐 API Configuration:', {
  baseURL: API_URL,
  environment: process.env.NODE_ENV
});

// Create axios instance
const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 10000, // 10 seconds timeout
});

// Request interceptor to add auth token and logging
api.interceptors.request.use(
  (config) => {
    const token = tokenStorage.getToken();
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    
    console.log(`📤 API Request: ${config.method?.toUpperCase()} ${config.url}`, {
      headers: config.headers,
      data: config.data ? '***' : null, // Don't log sensitive data
      hasAuth: !!token
    });
    
    return config;
  },
  (error) => {
    console.error('📤 Request Error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor to handle errors and logging
api.interceptors.response.use(
  (response) => {
    console.log(`📥 API Response: ${response.config.method?.toUpperCase()} ${response.config.url}`, {
      status: response.status,
      statusText: response.statusText,
      hasData: !!response.data
    });
    return response;
  },
  async (error) => {
    const originalRequest = error.config;

    console.error(`📥 API Error: ${originalRequest?.method?.toUpperCase()} ${originalRequest?.url}`, {
      status: error.response?.status,
      statusText: error.response?.statusText,
      data: error.response?.data,
      message: error.message
    });

    if (error.response?.status === 401 && !originalRequest._retry) {
      originalRequest._retry = true;
      
      console.log('🔄 Token expired, clearing auth and redirecting to login');
      
      // Clear auth and redirect to login
      tokenStorage.clearAuth();
      
      // Only redirect if we're not already on the login page
      if (window.location.pathname !== '/login') {
        window.location.href = '/login';
      }
    }

    return Promise.reject(error);
  }
);

export default api;