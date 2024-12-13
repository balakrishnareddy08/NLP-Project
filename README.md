# Emotion Graph-Enhanced Response Generation

## Project Overview

Emotion Aware Response Generation is an advanced AI-driven conversational system designed to generate emotionally intelligent and contextually appropriate responses. By integrating cutting-edge emotion detection, graph-based context tracking, This project aims to revolutionize human-AI interactions with genuinely empathetic communication.

## Key Features

### Emotion Intelligence
- Advanced emotion detection
- Context-aware response generation
- Dynamic emotional state tracking

### Technical Innovations
- Fine-tuned LLaMA language model
- Custom emotional graph database
- Multi-task learning framework
- Parameter-Efficient Fine-tuning (LoRA)

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended) with atleast 8GB
- 32GB+ RAM

### Setup
```bash
# Clone the repository
git clone https://github.com/balakrishnareddy08/NLP-Project.git
cd Code

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Unix/macOS
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Running the Application
```bash
streamlit run app.py
```

## Project Structure
```
NLP_Project/Code
│
├── data/           # Datasets and preprocessing
├── configs/         # Model configurations
├── modules/        # Core system modules
│   ├── emotion_detection/
│   ├── graph_processor/
│   └── response_generator/
├── tests/          # Unit and integration tests
├── interface/      # Streamlit UI
├── utils/      # Managing Utility functions
└── requirements.txt
```

## Technical Architecture

### Core Components
- Emotion Detection Engine
- Graph-Based Context Tracker
- Preference Learning System
- Streamlit Interactive Interface

## Development Roadmap

### Current Capabilities
- Text-based emotional intelligence
- Contextual response generation
- Emotion graph tracking

### Future Enhancements
- [ ] Multi-modal emotion detection
- [ ] Advanced preference learning
- [ ] Scalability improvements
- [ ] Multi-language support

## Contributing

1. Fork the repository
2. Create a feature branch
   ```bash
   git checkout -b feature/amazing-improvement
   ```
3. Commit your changes
   ```bash
   git commit -m 'Add some amazing improvement'
   ```
4. Push to the branch
   ```bash
   git push origin feature/amazing-improvement
   ```
5. Open a Pull Request

## Team

**Project Contributors:**
- **Bala krishna Ragannagari**
- **Bhagavath Sai Darapureddy**

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Acknowledgments

- EmpatheticDialogues Dataset
- LLaMA Model
- Hugging Face Transformers


