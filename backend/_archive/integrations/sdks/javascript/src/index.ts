/**
 * M&A Intelligence Platform SDK for JavaScript/TypeScript
 * 
 * @example Basic Usage
 * ```typescript
 * import { MAIntelligenceClient } from '@ma-intelligence/sdk';
 * 
 * const client = new MAIntelligenceClient({
 *   apiKey: 'ak_your_api_key',
 *   baseUrl: 'https://api.ma-intelligence.com'
 * });
 * 
 * // List companies
 * const companies = await client.companies.list({ city: 'Paris', caMin: 100000 });
 * 
 * // Search companies
 * const results = await client.companies.search({
 *   q: 'comptable',
 *   ville: 'Paris',
 *   withEmail: true
 * });
 * 
 * // Get company details
 * const company = await client.companies.get('company-id');
 * ```
 * 
 * @version 1.0.0
 * @author M&A Intelligence Team
 */

export { MAIntelligenceClient } from './client';
export * from './types';
export * from './errors';
export * from './models';