# Recall AI and Samsung AI
**On-Device AI-Powered Recall System**

---

## Overview
Recall AI is a mobile-first system that captures screenshots, performs OCR, filters sensitive data, generates embeddings, and stores them in a FAISS vector database with encryption. This enables RAG-based querying through a FastAPI backend. Samsung AI, on the other hand, is its on-device counterpart, offering the same pipeline of OCR, sensitive data filtering, embeddings, vector storage, and LLM-powered recall but entirely offline using TFLite models.

---

## Features
- **Mobile-First Design**: Capture screenshots and perform OCR on-device.
- **Sensitive Data Filtering**: Filter out sensitive information from captured text.
- **Embeddings Generation**: Generate embeddings from filtered text for efficient storage and querying.
- **Vector Database Storage**: Store embeddings in a FAISS vector database with encryption for secure and efficient querying.
- **LLM-Powered Recall**: Utilize LLMs for recall-based querying, providing accurate and relevant results.
- **Offline Capability**: Samsung AI operates entirely offline, ensuring zero API calls, complete privacy, and independence from external servers.

---

## Technology Stack
- **Backend Framework**: FastAPI
- **Database**: FAISS vector database
- **Encryption**: Utilizes encryption for secure storage and querying
- **Machine Learning Framework**: TensorFlow Lite (TFLite) for on-device inference
- **Programming Language**: Dart (for Samsung AI), Python (for Recall AI)

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Madhur-Prakash/Samsung-AI.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Samsung-AI
   ```
3. Install dependencies:
   ```bash
   flutter pub get
   ```
4. Set up the environment:
   - Ensure you have the necessary dependencies installed, including Flutter and Dart.

---

## Usage

1. Run the application:
   ```bash
   flutter run
   ```
2. Capture a screenshot and perform OCR:
   - Utilize the in-app functionality to capture a screenshot and perform OCR.
3. Query the vector database:
   - Use the LLM-powered recall functionality to query the vector database and retrieve relevant results.

---

## API Endpoints
Not applicable for Samsung AI, as it operates entirely offline. Recall AI's API endpoints are available through its FastAPI backend.

---

## Future Enhancements
- **Improve OCR Accuracy**: Enhance the OCR pipeline to improve text recognition accuracy.
- **Expand Model Support**: Add support for more machine learning models to improve recall accuracy and efficiency.
- **Enhance Security**: Implement additional security measures to protect user data and prevent unauthorized access.

---

## Contribution Guidelines

Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes and submit a pull request.

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Author
**Madhur-Prakash**  
[GitHub](https://github.com/Madhur-Prakash)