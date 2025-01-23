# Docbot - AI-Powered Document Assistant 🤖

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![OpenAI](https://img.shields.io/badge/OpenAI-412991.svg)](https://openai.com/)
[![FAISS](https://img.shields.io/badge/FAISS-0467C7.svg)](https://github.com/facebookresearch/faiss)

## Overview 🔍
Docbot is an intelligent document assistant that processes and analyzes documents to provide accurate answers to user queries. It leverages multiple advanced language models and retrieval techniques to ensure high-quality responses.

## Features ⭐
- Multi-document processing and analysis
- Multi-route retrieval system for improved accuracy
- Advanced reranking mechanism
- Interactive chat interface
- Support for various document formats
- Streaming responses with real-time feedback

## Technical Architecture 🏗️
### Retrieval System 🔍
- **Multiple Embedding Models**:
  - GTE-large-zh: Optimized for Chinese text understanding
  - BGE-large-zh: Enhanced semantic comprehension
  - BM25: Classical information retrieval algorithm
- **Reranking**: Uses BGE-reranker-large for context optimization
- **Large Language Model**: Powered by GPT for natural language generation

## How It Works 🛠️
1. **Document Processing** 📄:
   - Documents are loaded and split into manageable chunks
   - Each chunk is processed through multiple embedding models

2. **Query Processing** 🔎:
   - User queries are processed through multiple retrieval routes
   - Results are combined and reranked for relevance
   - Most relevant context is selected for the final response

3. **Response Generation** 💬:
   - Selected context is combined with the user query
   - LLM generates natural and accurate responses
   - Responses are streamed in real-time

## Installation & Usage Guide 🚀

### Prerequisites 📋
- Python 3.10+
- CUDA-capable GPU (recommended)
- [uv](https://github.com/astral-sh/uv) package manager (recommended)
- OpenAI API key

### Installation Steps 📥
1. **Clone Repository**
```bash
git clone https://github.com/AbyssSkb/Docbot
cd Docbot
```

2. **Install Dependencies**
```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -r requirements.txt
```

3. **Environment Setup**
- Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=your_base_url  # Optional
OPENAI_LLM_MODEL=your_preferred_model  # Default: gpt-4o
```

4. **Document Setup**
- Create a `doc` folder in the project root
- Place your documents in the `doc` folder
- Generate document indexes:
```bash
python create_index.py
```

### Running the Application 🏃
```bash
streamlit run main.py
```

### Basic Usage 💡
1. Open the provided URL in your web browser
2. Enter your questions in the chat interface
3. View real-time responses based on your documents

## Limitations and Considerations ⚠️
1. **Language Support** 🌐:
   - Primary optimization for Chinese text
   - English support can be enabled by switching to English-language models
   - Consider language-specific requirements for your use case

2. **Text Processing** 📝:
   - Jieba tokenizer is optimized for Chinese
   - Basic English tokenization support
   - May require adjustment for other languages

3. **Document Compatibility** 📄:
   - Uses LangChain's DirectoryLoader
   - Some document formats may have compatibility issues
   - Verify support for your specific document types

## Contributing 🤝
Contributions are welcome! Please feel free to submit pull requests or create issues for bugs and feature requests.

## License ⚖️
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

