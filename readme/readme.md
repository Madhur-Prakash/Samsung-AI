# Samsung EnnovateX 2025 AI Challenge Submission

## Problem Statement
> Reimagine a smartphone that doesn't just run apps, but truly understands and assists users. An agent that sees what you see, hears what you hear, and remembers your experiences to provide contextual, real-time help, all without a constant connection to the cloud.

## Team Name
**Manusmriti**

## Team Members
- **Member 1**: Pranav Sharma
- **Member 2**: Nidhi Singh  
- **Member 3**: Madhur Prakash Mangal
- **Member 4**: Nandika Gupta

## Demo Video Link
[YouTube Demo](https://youtube.com/watch?v=demo-link)

---

# Project Artefacts

## Technical Documentation
[Technical Docs](maindocs.md)  
*All technical details are documented in markdown files, including system architecture, implementation details, performance metrics, and deployment instructions.*

## Source Code
[GitHub Repository](https://github.com/your-repo)  
*Complete Flutter application source code with all modules, dependencies, and build configurations for successful installation and execution on Android platforms.*

## Models Used
All models utilized are **open-source** and publicly available:

| Model | Category | License | Purpose | Key Features | Performance | Repository |
|-------|----------|---------|---------|--------------|-------------|------------|
| **[YoloV8s-oiv7 Small](https://huggingface.co/Ultralytics/YOLOv8)** | Core AI Models | GPL-3.0 | Real-time object and action detection | Fine-tuned on UCF101 dataset for 5 action categories | <100ms inference time on mobile devices | [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) |
| **[MobileNet v2](https://huggingface.co/docs/transformers/en/model_doc/mobilenet_v2)** | Core AI Models | Apache 2.0 | Efficient video processing backbone | TensorFlow Lite quantized for mobile deployment | <50MB model size | [TensorFlow Models](https://github.com/tensorflow/models) |
| **[Vosk API](https://github.com/alphacep/vosk-api)** | Speech Processing | Apache 2.0 | Offline speech-to-text recognition | 20+ language models supported, complete offline operation | Real-time streaming recognition with zero latency, 50MB model size | [Vosk API](https://github.com/alphacep/vosk-api) |
| **[OLLAMA all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)** | Natural Language Processing | Apache 2.0 | Sentence embeddings for RAG implementation | PyTorch → TensorFlow Lite conversion, 384-dimensional embeddings | <50ms inference per sentence on mobile CPU | [Sentence Transformers](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) |

> *Note: YoloV8s-oiv7 offers dual licensing—AGPL-3.0 for open-source use and an Enterprise License for commercial applications.*

## Models Published 

| Model | Category | License | Purpose | Key Features | Performance | Parent Model |
|-------|----------|---------|---------|--------------|-------------|---------------------|
| **[distilgpt2](https://huggingface.co/Madhur-Prakash-Mangal/distilgpt2-TFLITE)** | Core AI Models | Apache 2.0 | Text generation and completion | Smaller, faster version of DistilGPT-2 | <50ms inference time on mobile devices | [Hugging Face](https://huggingface.co/distilgpt2) |
| **[Embedding Model](https://huggingface.co/Madhur-Prakash-Mangal/all-MiniLM-L6-v2-tflite)** | Core AI Models | Apache 2.0 | Text embedding generation | Trained on diverse text corpora | <50ms inference time on mobile devices | [Hugging Face](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) |
| **[Action Detection Model](https://huggingface.co/NanG01/Action_detection)** | Core AI Models | Apache 2.0 | Customized action detection | Trained on UCF101 for 5 action categories | <100ms inference time on mobile devices | [Hugging Face](https://huggingface.co/docs/transformers/en/model_doc/mobilenet_v2) |

## Datasets Used
*All datasets are publicly available under open licenses:*

| **Dataset** | **License** | **Purpose** | **Size** | **Usage** | **Source/Provider** |
|-------------|-------------|-------------|----------|-----------|----------------------|
| [UCF101 Action Recognition](https://www.kaggle.com/datasets/abdallahwagih/ucf101-videos) | Creative Commons License | Training YoloV8s-oiv7 for action detection | 13,320 videos across 101 categories | 5 selected categories for mobile optimization | University of Central Florida |
| [Open Images v7](https://storage.googleapis.com/openimages/web/index.html) | Creative Commons BY 4.0 | Pre-training foundation for YoloV8s-oiv7 object detection | 9M+ images with detailed annotations | Indirect - model pre-training foundation | Google Research |

---

# Attribution

This project builds upon several **open-source projects** while contributing novel innovations:

## Core Open Source Dependencies

### Mobile Development Framework
- **[Flutter](https://github.com/flutter/flutter)** *(Open Source - BSD 3-Clause)*
  - **Usage**: Cross-platform mobile application development
  - **Version**: Latest stable release
  - **License**: BSD 3-Clause License

### Audio/Video Processing
- **[FFmpeg](https://github.com/FFmpeg/FFmpeg)** *(Open Source - LGPL/GPL)*
  - **Usage**: Real-time audio extraction from video streams
  - **Integration**: Flutter plugin for seamless audio processing
  - **License**: Lesser General Public License (LGPL)

### AI/ML Frameworks
| **Dependency** | **Usage / Integration** | **Provider** | **License** |
|----------------|--------------------------|--------------|-------------|
| [TensorFlow Lite](https://github.com/tensorflow/tensorflow) | On-device model inference & optimization (Custom TFLite interpreter) | Google | Apache 2.0 |
| [ai-edge-torch](https://github.com/google-ai-edge/ai-edge-torch) | Converts PyTorch models to mobile-optimized TFLite | Google AI Edge Team | Apache 2.0 |



### Flutter Ecosystem
| **Dependency** | **Usage / Integration** | **License** |
|----------------|--------------------------|-------------|
| [go_router](https://pub.dev/packages/go_router) | Declarative routing & navigation | BSD 3-Clause |
| [flutter_riverpod](https://pub.dev/packages/flutter_riverpod) | Reactive state management & dependency injection | MIT |

---

# License & Acknowledgments

## Open Source Licenses
- **Flutter Framework**: BSD 3-Clause License
- **YoloV8s-oiv7**: GPL-3.0 License  
- **Vosk API**: Apache 2.0 License
- **FFmpeg**: LGPL/GPL License
- **TensorFlow Lite**: Apache 2.0 License
- **ai-edge-torch**: Apache 2.0 License
- **MobileNet v2**: Apache 2.0 License

## Dataset Licenses
- **UCF101**: Creative Commons License
- **Open Images**: Creative Commons BY 4.0

## Special Acknowledgments

### Industry Partners
- **Samsung Research**: For organizing the EnnovateX 2025 AI Challenge and providing the vision for on-device personal AI assistants
- **Google AI Edge**: For developing and open-sourcing ai-edge-torch conversion tools that enable mobile AI deployment
- **Ultralytics**: For the YoloV8s-oiv7 framework, comprehensive documentation, and active community support

### Academic Contributors  
- **University of Central Florida**: For creating and maintaining the UCF101 action recognition dataset
- **Google Research**: For the Open Images dataset that serves as foundation training data

### Open Source Communities
- **Flutter Community**: For exceptional documentation, packages, and developer support ecosystem
- **TensorFlow Community**: For mobile AI frameworks and optimization tools
- **Vosk Community**: For offline speech recognition models and multi-language support

---

*Built with ❤️ for Samsung EnnovateX 2025 AI Challenge*  
*Advancing Privacy-First On-Device AI for the Next Generation of Mobile Intelligence*




## Novel Features Developed

### 🎙️ **Intelligent Speech Monitoring System**
- **Real-time Speech-to-Text**: Continuous offline speech capture using Vosk
- **Privacy-First Architecture**: End-to-end encryption with no cloud dependencies
- **Tokenized Storage**: Secure chunking and encrypted file system storage

### 🎯 **Advanced Action Detection Pipeline**
- **Multi-Modal Fusion**: YoloV8s-oiv7 + MobileNet v2 integration for comprehensive action understanding
- **Context Generation**: Automatic text file creation from detected visual actions
- **Mobile Optimization**: Custom quantization for real-time mobile inference

### 🧠 **On-Device RAG Implementation**
- **Local Vector Database**: SQLite-based semantic search with encrypted embeddings
- **Privacy-Preserving AI**: Complete offline operation with zero data transmission
- **Contextual Intelligence**: LLM-powered responses based solely on personal user data

### 🔐 **Samsung Knox Integration Ready**
- **Enterprise Security**: Built-in encryption aligned with Samsung security standards
- **Edge Computing**: Maximizes device capabilities while protecting user privacy
- **Zero API Dependencies**: Complete functionality without internet connectivity

## Innovation Highlights

### Technical Achievements
- **First Mobile RAG**: Novel implementation of Retrieval Augmented Generation entirely on-device
- **Multi-Modal AI Fusion**: Seamless integration of speech, vision, and text processing
- **Privacy by Design**: Built-in encryption and local-only processing architecture
- **Edge AI Optimization**: Advanced model quantization and mobile-specific optimizations

### Samsung EnnovateX Alignment
- **On-Device Processing**: 100% offline functionality with zero cloud dependencies
- **Privacy Leadership**: No personal data leaves the device, ever
- **Real-Time Intelligence**: Immediate response without network latency
- **Samsung Ecosystem Ready**: Optimized for Galaxy devices and Knox security framework


