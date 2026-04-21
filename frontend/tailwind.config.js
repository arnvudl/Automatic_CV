/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        brand: { DEFAULT: "#3B82F6", dark: "#1D4ED8", light: "rgba(59,130,246,0.15)" },
        glass: {
          white:  "rgba(255,255,255,0.08)",
          border: "rgba(255,255,255,0.14)",
        },
      },
      backdropBlur: {
        xs: '4px',
      },
      boxShadow: {
        glass: '0 8px 32px rgba(0,0,0,0.28)',
        glow:  '0 0 24px rgba(59,130,246,0.45)',
      },
      borderRadius: {
        '2.5xl': '20px',
      },
    },
  },
  plugins: [],
}
