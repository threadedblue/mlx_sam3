# SAM3 Segmentation Studio

An interactive web application for image segmentation using the SAM3 model. Features a modern Next.js frontend with a FastAPI backend.

## Features

- **Text Prompts**: Describe what you want to segment (e.g., "person", "dog", "car")
- **Box Prompts**: Draw bounding boxes to include or exclude regions
- **Real-time Visualization**: See segmentation masks and bounding boxes overlaid on your image
- **Session Management**: Multiple users can use the app simultaneously

## Architecture

```
app/
├── backend/           # FastAPI server
│   ├── main.py       # API endpoints
│   └── requirements.txt
└── frontend/          # Next.js app
    └── src/
        ├── app/       # Pages and layout
        ├── components/  # React components
        └── lib/       # API client and utilities
```

## Prerequisites

- Python 3.10+ with the SAM3 model dependencies installed
- Node.js 18+
- SAM3 model weights at `sam3-mod-weights/model.safetensors`

## Quick Start

Run both servers with a single command:

```bash
cd app
./run.sh
```

This will start:
- **Backend**: http://localhost:8000
- **Frontend**: http://localhost:3000
- **API Docs**: http://localhost:8000/docs

Press `Ctrl+C` to stop both servers.

---

## Manual Setup

### Backend

1. Install Python dependencies:

```bash
cd app/backend
pip install -r requirements.txt
```

2. Start the backend server:

```bash
python main.py
# Or with uvicorn directly:
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The backend will load the SAM3 model on startup (this may take a minute).

### Frontend

1. Install Node dependencies:

```bash
cd app/frontend
npm install
```

2. Start the development server:

```bash
npm run dev
```

3. Open [http://localhost:3000](http://localhost:3000) in your browser

## Usage

1. **Upload an Image**: Click the upload area or drag & drop an image
2. **Text Segmentation**: Enter a text prompt like "apple" or "face" and click the segment button
3. **Box Prompts**: 
   - Select "Include" to draw boxes around objects you want to segment
   - Select "Exclude" to draw boxes around regions to exclude
   - Draw by clicking and dragging on the image
4. **Reset**: Click "Clear All Prompts" to start fresh with the same image

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Check backend status |
| `/upload` | POST | Upload image and create session |
| `/segment/text` | POST | Segment with text prompt |
| `/segment/box` | POST | Add box prompt |
| `/reset` | POST | Reset all prompts |
| `/session/{id}` | DELETE | Delete session |

## Environment Variables

### Frontend

- `NEXT_PUBLIC_API_URL`: Backend API URL (default: `http://localhost:8000`)

## Development

### Frontend

```bash
cd app/frontend
npm run dev      # Development server
npm run build    # Production build
npm run lint     # Lint code
```

### Backend

```bash
cd app/backend
uvicorn main:app --reload  # Hot reload during development
```

## Tech Stack

- **Frontend**: Next.js 15, React 19, Tailwind CSS 4, TypeScript
- **Backend**: FastAPI, Uvicorn
- **Model**: SAM3 (MLX implementation)

