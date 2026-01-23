# RAG Application - AI Analyst & Document Ranker

A comprehensive RAG (Retrieval-Augmented Generation) application built with Streamlit, featuring AI-powered document analysis and resume ranking capabilities.

## 🚀 Features

### AI Analyst (`ai_analyst.py`)
- **Interactive Q&A Interface**: Chat with your documents using Google Gemini AI
- **Multi-format Support**: Upload and analyze PDF, TXT, and DOCX files
- **Vector Search**: Powered by FAISS for efficient document retrieval
- **Chat History**: Persistent conversation storage
- **Real-time Processing**: Instant document analysis and response generation

### Document Ranker (`document_ranker.py`)
- **Resume Ranking**: Intelligent matching of resumes to job descriptions
- **Batch Processing**: Rank multiple resumes simultaneously
- **Similarity Scoring**: Advanced text similarity algorithms
- **Export Results**: Download ranked results as CSV

### RAG Engine (`rag_app.py`)
- **Vector Database Integration**: FAISS and Milvus support
- **Text Chunking**: Intelligent document segmentation
- **Embedding Generation**: Google Generative AI embeddings
- **Retrieval System**: Context-aware document retrieval

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **AI/ML**: Google Generative AI (Gemini), LangChain
- **Vector Databases**: FAISS, Milvus
- **Document Processing**: PyPDF, python-docx
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Aditya1517/GenAI-Notes-Codes.git
   cd "GenAI-Notes-Codes/LangChain/Interview Task/RAG Application"
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # or
   source .venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   Create a `.env` file with your Google API key:
   ```
   GOOGLE_API_KEY=your_google_api_key_here
   ```

5. **Start Milvus (Optional)**:
   ```bash
   docker-compose up -d
   ```

## 🚀 Usage

### Running the AI Analyst
```bash
streamlit run ai_analyst.py
```

### Running the Document Ranker
```bash
streamlit run document_ranker.py
```

### Using the RAG Engine
```python
from rag_app import RAGApplication

# Initialize RAG app
rag = RAGApplication()

# Process documents
rag.process_documents("path/to/documents")

# Query documents
response = rag.query("Your question here")
```

## 📁 Project Structure

```
RAG Application/
├── ai_analyst.py           # Main Streamlit AI analyst app
├── document_ranker.py      # Resume ranking application
├── rag_app.py             # Core RAG implementation
├── requirements.txt       # Python dependencies
├── docker-compose.yml     # Milvus setup
├── .streamlit/
│   └── config.toml       # Streamlit configuration
├── .env                  # Environment variables (create this)
└── README.md            # This file
```

## 🔧 Configuration

### Streamlit Configuration
The app uses custom Streamlit configuration in `.streamlit/config.toml` for optimal performance and appearance.

### Vector Database
- **FAISS**: Local vector storage for quick prototyping
- **Milvus**: Scalable vector database for production use

## 🌟 Key Features Explained

### 1. Document Processing
- Automatic text extraction from multiple formats
- Intelligent chunking with overlap for context preservation
- Metadata preservation for source tracking

### 2. Vector Search
- Semantic similarity search using embeddings
- Configurable similarity thresholds
- Multi-document retrieval and ranking

### 3. AI Integration
- Google Gemini Pro for advanced language understanding
- Context-aware response generation
- Conversation memory for coherent dialogues

### 4. User Interface
- Clean, intuitive Streamlit interface
- Real-time feedback and progress indicators
- Responsive design for various screen sizes

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is part of the GenAI-Notes-Codes repository. Please refer to the main repository for licensing information.

## 🙏 Acknowledgments

- Google Generative AI for powerful language models
- LangChain for RAG framework
- Streamlit for the amazing web framework
- The open-source community for various tools and libraries

## 📞 Support

For questions or support, please open an issue in the GitHub repository or contact the maintainer.

---

**Happy Coding! 🚀**