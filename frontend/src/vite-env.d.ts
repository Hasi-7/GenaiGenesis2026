/// <reference types="vite/client" />

interface Window {
  cognitiveSense: {
    notify: (title: string, body: string) => Promise<void>;
    show: () => Promise<void>;
    hide: () => Promise<void>;
  };
}
