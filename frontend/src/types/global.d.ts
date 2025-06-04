/**
 * Déclarations de types globaux pour les modules externes
 */

// Google Analytics gtag
declare module 'gtag' {
  interface GtagConfig {
    page_title?: string;
    page_location?: string;
    [key: string]: any;
  }

  interface GtagEvent {
    event_category?: string;
    event_label?: string;
    value?: number;
    [key: string]: any;
  }

  function gtag(command: 'config', targetId: string, config?: GtagConfig): void;
  function gtag(command: 'event', eventName: string, eventParameters?: GtagEvent): void;
  function gtag(command: 'set', config: { [key: string]: any }): void;
  function gtag(command: string, ...args: any[]): void;

  export = gtag;
}

// Global gtag function
declare global {
  function gtag(command: 'config', targetId: string, config?: any): void;
  function gtag(command: 'event', eventName: string, eventParameters?: any): void;
  function gtag(command: 'set', config: { [key: string]: any }): void;
  function gtag(command: string, ...args: any[]): void;

  interface Window {
    gtag?: typeof gtag;
  }
}

// Mixpanel Browser
declare module 'mixpanel-browser' {
  interface MixpanelLib {
    init(token: string, config?: any): void;
    track(event: string, properties?: any): void;
    identify(distinctId: string): void;
    people: {
      set(properties: any): void;
      set_once(properties: any): void;
      increment(properties: any): void;
    };
    register(properties: any): void;
    unregister(property: string): void;
    reset(): void;
    get_distinct_id(): string;
    alias(alias: string, distinctId?: string): void;
    time_event(event: string): void;
    [key: string]: any;
  }

  const mixpanel: MixpanelLib;
  export default mixpanel;
}

export {};