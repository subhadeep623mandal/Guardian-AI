# Requirements

## System Requirements

| Requirement | Minimum Version | Recommended |
|---|---|---|
| **Node.js** | v18.0.0 | v20+ (LTS) |
| **npm** | v8.0.0 | v10+ |
| **OS** | Windows 10 / macOS 11 / Ubuntu 20.04 | Latest |
| **Browser** | Chrome 90+ / Firefox 88+ / Edge 90+ | Chrome latest |
| **RAM** | 4 GB | 8 GB |
| **Webcam** | Optional | Required for biometric login screen |

---

## API Requirements

| Service | Required | Purpose |
|---|---|---|
| **Google Gemini API Key** | ✅ Yes | Image analysis + heatmap generation |

### Getting a Gemini API Key
1. Go to [https://aistudio.google.com/apikey](https://aistudio.google.com/apikey)
2. Sign in with your Google account
3. Click **"Create API key"**
4. Copy the key and paste it into your `.env.local` file

### Free Tier Limits
| Model | Requests/Minute | Requests/Day |
|---|---|---|
| `gemini-2.0-flash` | 15 RPM | 1,500 RPD |
| `gemini-2.0-flash-lite` | 30 RPM | 1,500 RPD |
| `gemini-2.5-flash` | 10 RPM | 500 RPD |
| `gemini-2.0-flash-exp-image-generation` | 10 RPM | 1,000 RPD |

---

## Node.js Dependencies

### Production Dependencies

| Package | Version | Purpose |
|---|---|---|
| `@google/genai` | ^1.29.0 | Google Gemini AI SDK |
| `react` | ^19.0.0 | UI framework |
| `react-dom` | ^19.0.0 | React DOM renderer |
| `motion` | ^12.23.24 | Animations (Framer Motion) |
| `lucide-react` | ^0.546.0 | Icon library |
| `react-dropzone` | ^15.0.0 | Drag-and-drop file upload |
| `clsx` | ^2.1.1 | Conditional class names |
| `tailwind-merge` | ^3.5.0 | Tailwind class merging |
| `@tailwindcss/vite` | ^4.1.14 | Tailwind CSS v4 Vite plugin |
| `vite` | ^6.2.0 | Build tool and dev server |
| `dotenv` | ^17.2.3 | Environment variable loading |
| `express` | ^4.21.2 | Server (if using SSR/proxy) |

### Development Dependencies

| Package | Version | Purpose |
|---|---|---|
| `typescript` | ~5.8.2 | Type checking |
| `@vitejs/plugin-react` | ^5.0.4 | React Fast Refresh in Vite |
| `tailwindcss` | ^4.1.14 | CSS utility framework |
| `autoprefixer` | ^10.4.21 | CSS vendor prefixes |
| `tsx` | ^4.21.0 | TypeScript execution |
| `@types/node` | ^22.14.0 | Node.js type definitions |
| `@types/express` | ^4.17.21 | Express type definitions |

---

## Installation

```bash
# Verify your Node.js version
node --version   # Should be >= 18.0.0
npm --version    # Should be >= 8.0.0

# Install all dependencies from package.json
npm install

# Set up environment variables
cp .env.example .env.local
# Edit .env.local and add your GEMINI_API_KEY

# Start development server
npm run dev
```

## Available Scripts

| Script | Command | Description |
|---|---|---|
| **Dev server** | `npm run dev` | Starts Vite dev server at localhost:3000 |
| **Build** | `npm run build` | Builds production bundle to `dist/` |
| **Preview** | `npm run preview` | Preview production build locally |
| **Type check** | `npm run lint` | Run TypeScript type checking |
| **Clean** | `npm run clean` | Remove `dist/` directory |

---

## Environment Variables

Copy `.env.example` to `.env.local` and fill in your values:

```env
# Required: Your Gemini API key from https://aistudio.google.com/apikey
GEMINI_API_KEY="YOUR_GEMINI_API_KEY_HERE"

# Optional: URL where this app is hosted (used for production deployments)
APP_URL="http://localhost:3000"
```

> ⚠️ **Never commit `.env.local` to Git.** It is already excluded by `.gitignore`.
