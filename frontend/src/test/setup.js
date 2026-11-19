import '@testing-library/jest-dom/vitest';
import { afterEach } from 'vitest';
import { cleanup } from '@testing-library/react';

afterEach(() => {
  cleanup();
  window.history.replaceState(null, '', '/');
});

if (!URL.createObjectURL) URL.createObjectURL = () => 'blob:preview';
if (!URL.revokeObjectURL) URL.revokeObjectURL = () => {};
window.scrollTo = () => {};
