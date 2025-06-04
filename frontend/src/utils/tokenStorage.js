// Token storage utility with expiration and basic obfuscation
// TODO: Migrate to httpOnly cookies on backend for production security
// NOTE: Install crypto-js for production: npm install crypto-js

const TOKEN_KEY = 'auth_token';
const USER_KEY = 'user_data';
const EXPIRY_KEY = 'token_expiry';

class SecureTokenStorage {
  constructor() {
    // Simple base64 encoding for basic obfuscation (not secure encryption)
    // In production, use proper encryption with crypto-js
  }

  // Basic encoding (NOT secure encryption - just obfuscation)
  encode(data) {
    try {
      return btoa(JSON.stringify(data));
    } catch (error) {
      console.error('Encoding error:', error);
      return null;
    }
  }

  // Basic decoding
  decode(encodedData) {
    try {
      return JSON.parse(atob(encodedData));
    } catch (error) {
      console.error('Decoding error:', error);
      return null;
    }
  }

  // Set token with expiration
  setToken(token, expiresInMinutes = 30) {
    try {
      const encodedToken = this.encode(token);
      const expiryTime = new Date().getTime() + (expiresInMinutes * 60 * 1000);
      
      sessionStorage.setItem(TOKEN_KEY, encodedToken);
      sessionStorage.setItem(EXPIRY_KEY, expiryTime.toString());
      
      return true;
    } catch (error) {
      console.error('Error setting token:', error);
      return false;
    }
  }

  // Get token if not expired
  getToken() {
    try {
      const encodedToken = sessionStorage.getItem(TOKEN_KEY);
      const expiryTime = sessionStorage.getItem(EXPIRY_KEY);
      
      if (!encodedToken || !expiryTime) {
        return null;
      }

      // Check if token has expired
      if (new Date().getTime() > parseInt(expiryTime)) {
        this.clearAuth();
        return null;
      }

      return this.decode(encodedToken);
    } catch (error) {
      console.error('Error getting token:', error);
      this.clearAuth();
      return null;
    }
  }

  // Set user data
  setUserData(userData) {
    try {
      const encodedData = this.encode(userData);
      sessionStorage.setItem(USER_KEY, encodedData);
      return true;
    } catch (error) {
      console.error('Error setting user data:', error);
      return false;
    }
  }

  // Get user data
  getUserData() {
    try {
      const encodedData = sessionStorage.getItem(USER_KEY);
      if (!encodedData) {
        return null;
      }
      return this.decode(encodedData);
    } catch (error) {
      console.error('Error getting user data:', error);
      return null;
    }
  }

  // Check if token exists and is valid
  isAuthenticated() {
    return this.getToken() !== null;
  }

  // Clear all auth data
  clearAuth() {
    sessionStorage.removeItem(TOKEN_KEY);
    sessionStorage.removeItem(USER_KEY);
    sessionStorage.removeItem(EXPIRY_KEY);
    
    // Also clear old localStorage data for migration
    localStorage.removeItem('token');
    localStorage.removeItem('user');
  }

  // Get remaining time until expiration (in minutes)
  getTimeUntilExpiry() {
    try {
      const expiryTime = sessionStorage.getItem(EXPIRY_KEY);
      if (!expiryTime) {
        return 0;
      }
      
      const remaining = parseInt(expiryTime) - new Date().getTime();
      return Math.max(0, Math.floor(remaining / (1000 * 60)));
    } catch (error) {
      return 0;
    }
  }
}

// Create singleton instance
const tokenStorage = new SecureTokenStorage();

export default tokenStorage;

// Legacy migration helper
export const migrateLegacyStorage = () => {
  const legacyToken = localStorage.getItem('token');
  const legacyUser = localStorage.getItem('user');
  
  if (legacyToken && legacyUser) {
    try {
      tokenStorage.setToken(legacyToken);
      tokenStorage.setUserData(JSON.parse(legacyUser));
      
      // Clear legacy storage
      localStorage.removeItem('token');
      localStorage.removeItem('user');
      
      console.log('Legacy tokens migrated to secure storage');
    } catch (error) {
      console.error('Error migrating legacy tokens:', error);
    }
  }
};