import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'

if (import.meta.env.PROD) {
  const script = document.createElement('script');
  script.async = true;
  script.src = 'https://www.googletagmanager.com/gtag/js?id=G-8NM6L1X6ML';
  document.head.appendChild(script);

  window.dataLayer = window.dataLayer || [];
  window.gtag = function(...args: unknown[]) { window.dataLayer.push(args); };
  window.gtag('js', new Date());
  window.gtag('config', 'G-8NM6L1X6ML');
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
