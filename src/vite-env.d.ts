/// <reference types="vite/client" />

declare global {
  interface Window {
    dataLayer: unknown[];
  }
}

export {};
