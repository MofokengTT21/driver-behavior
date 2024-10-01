/** @type {import('tailwindcss').Config} */
module.exports = {
  mode: 'jit',
  content: [
    './ets2-telemetry-server/server/Html/skins/truck-monitor/**/*.html',
    './src/**/*.{js,jsx,ts,tsx}',
  ],
  theme: {
    extend: {
      colors: {
        customGray: '#303030',
      },
    },
  },
  plugins: [],
}
