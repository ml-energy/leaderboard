/// <reference types="vite/client" />

declare global {
  const __LAST_UPDATED__: string;
  interface Window {
    dataLayer: unknown[];
    gtag: (...args: unknown[]) => void;
  }
}

export {};
