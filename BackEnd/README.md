# Chatbot with RAG (Retrieval-Augmented Generation)

A FastAPI-based chatbot that can process documents and answer questions using Google's Gemini AI.

## Features

- Upload and process multiple file types (PDF, DOCX, CSV, Excel, Images)
- Extract text from images using OCR
- Create vector embeddings using Google's embedding model
- Chat with documents using RAG (Retrieval-Augmented Generation)
- Streaming responses for real-time interaction

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install Tesseract OCR (for image processing)

**Windows:**
1. Download Tesseract installer from: https://github.com/tesseract-ocr/tesseract
2. Install and add to PATH (usually `C:\Program Files\Tesseract-OCR`)

**Linux:**
```bash
sudo apt-get install tesseract-ocr
```

**macOS:**
```bash
brew install tesseract
```

### 3. Set Environment Variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_api_key_here
DEBUG=False
```

Get your Google API key from: https://makersuite.google.com/app/apikey

### 4. Run the Application

```bash
uvicorn main:app --reload
```

The server will start at `http://127.0.0.1:8000`

## API Endpoints

### POST /upload/
Upload documents for processing. Supports:
- PDF files (.pdf)
- Word documents (.docx)
- CSV files (.csv)
- Excel files (.xlsx)
- Images (.png, .jpg, .jpeg)

### POST /chat/
Send a query to chat with the uploaded documents.

Request body:
```json
{
    "query": "Your question here"
}
```

## Troubleshooting

### Common Issues:

1. **ModuleNotFoundError: No module named 'pytesseract'**
   - Install pytesseract: `pip install pytesseract`
   - Install Tesseract OCR engine (see setup instructions above)

2. **DefaultCredentialsError**
   - Make sure `GOOGLE_API_KEY` environment variable is set
   - Check that your API key is valid

3. **Import errors with langchain**
   - Update to latest versions: `pip install --upgrade langchain langchain-community langchain-google-genai`

## File Structure

```
Chatbot_clg/
├── main.py                 # FastAPI application
├── requirements.txt        # Python dependencies
├── README.md              # This file
└── utils/
    ├── file_readers.py    # File processing utilities
    ├── embedding.py       # Vector embedding functions
    └── rag.py            # RAG implementation with Gemini
```
