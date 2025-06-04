// Environment validation utility
// Validates required environment variables on app startup

const REQUIRED_ENV_VARS = [
  'REACT_APP_API_URL'
];

const OPTIONAL_ENV_VARS = [
  'REACT_APP_ENV'
];

class EnvValidator {
  constructor() {
    this.errors = [];
    this.warnings = [];
  }

  validate() {
    // Check required variables
    REQUIRED_ENV_VARS.forEach(varName => {
      const value = process.env[varName];
      if (!value || value.trim() === '') {
        this.errors.push(`Missing required environment variable: ${varName}`);
      } else {
        console.log(`✅ ${varName}: ${value}`);
      }
    });

    // Check optional variables and warn if missing
    OPTIONAL_ENV_VARS.forEach(varName => {
      const value = process.env[varName];
      if (!value || value.trim() === '') {
        this.warnings.push(`Optional environment variable not set: ${varName}`);
      } else {
        console.log(`ℹ️ ${varName}: ${value}`);
      }
    });

    // Validate API URL format
    const apiUrl = process.env.REACT_APP_API_URL;
    if (apiUrl) {
      try {
        new URL(apiUrl);
        if (!apiUrl.includes('/api/v1')) {
          this.warnings.push('REACT_APP_API_URL should include /api/v1 endpoint');
        }
      } catch (error) {
        this.errors.push(`Invalid REACT_APP_API_URL format: ${apiUrl}`);
      }
    }

    return {
      isValid: this.errors.length === 0,
      errors: this.errors,
      warnings: this.warnings
    };
  }

  logResults() {
    if (this.errors.length > 0) {
      console.error('❌ Environment validation failed:');
      this.errors.forEach(error => console.error(`  - ${error}`));
    }

    if (this.warnings.length > 0) {
      console.warn('⚠️ Environment warnings:');
      this.warnings.forEach(warning => console.warn(`  - ${warning}`));
    }

    if (this.errors.length === 0) {
      console.log('✅ Environment validation passed');
    }
  }

  throwIfInvalid() {
    const result = this.validate();
    this.logResults();
    
    if (!result.isValid) {
      throw new Error(`Environment validation failed: ${result.errors.join(', ')}`);
    }
    
    return result;
  }
}

// Create singleton instance
const envValidator = new EnvValidator();

export default envValidator;

// Convenience function for app startup
export const validateEnvironment = () => {
  return envValidator.throwIfInvalid();
};