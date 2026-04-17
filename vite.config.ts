import { defineConfig } from 'vite';
import vue from '@vitejs/plugin-vue';

export default defineConfig({
  plugins: [
    vue(),
    {
      name: 'exclude-wasm-assets',
      generateBundle(_, bundle) {
        for (const name of Object.keys(bundle)) {
          if (name.endsWith('.wasm')) {
            delete bundle[name];
          }
        }
      },
    },
  ],
  server: {
    host: '0.0.0.0',
    port: 5173,
  },
  worker: {
    format: 'es',
  },
});
