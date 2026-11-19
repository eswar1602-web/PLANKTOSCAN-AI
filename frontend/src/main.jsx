import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import '@fontsource/syne/700.css';
import '@fontsource/onest/400.css';
import '@fontsource/onest/500.css';
import '@fontsource/jetbrains-mono/500.css';
import './styles/app.css';
import App from './App.jsx';

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
