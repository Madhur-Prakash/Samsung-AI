import 'dart:io';
import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:samsung_ai/services/embeddings.dart';
import 'package:samsung_ai/services/vector_store.dart';
import 'package:tiktoken/tiktoken.dart';
import 'services/crypto_service.dart';
import 'dart:math' show exp, Random;
import 'services/pipeline.dart';
import 'services/performance_monitor.dart';
import 'services/debug_utils.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Crypto & Embedding Demo',
      theme: ThemeData(primarySwatch: Colors.blue),
      home: const HomeScreen(),
    );
  }
}

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  String _status = "Loading models...";
  final TextEditingController _chatController = TextEditingController();
  String _chatResponse = "";

  late Interpreter _interpreter;
  bool _isModelLoaded = false;
  final enc = getEncoding('gpt2');
  
  // Cache embeddings model to avoid reloading
  Embeddings? _cachedEmbeddings;
  bool _isEmbeddingsLoaded = false;

  @override
  void initState() {
    super.initState();
    _loadModels();
  }
  
  @override
  void dispose() {
    _chatController.dispose();
    // Clean up interpreter if needed
    try {
      _interpreter.close();
    } catch (e) {
      print("Error closing interpreter: $e");
    }
    PerformanceMonitor.printSummary();
    super.dispose();
  }

  Future<void> _loadModels() async {
    try {
      PerformanceMonitor.startTimer("Load LLM Model");
      PerformanceMonitor.logMemoryUsage("Before loading models");
      
      // Load LLM model
      _interpreter = await Interpreter.fromAsset("assets/models/distilgpt2.tflite");
      PerformanceMonitor.endTimer("Load LLM Model");
      debugPrint("✅ DistilGPT-2 model loaded!");
      
      setState(() {
        _isModelLoaded = true;
        _status = "Loading embeddings model...";
      });
      
      // Load and cache embeddings model
      PerformanceMonitor.startTimer("Load Embeddings Model");
      _cachedEmbeddings = await Embeddings.load();
      PerformanceMonitor.endTimer("Load Embeddings Model");
      
      PerformanceMonitor.startTimer("Validate Embeddings Model");
      final isValid = await _cachedEmbeddings!.validateModel();
      PerformanceMonitor.endTimer("Validate Embeddings Model");
      
      PerformanceMonitor.logMemoryUsage("After loading models");
      
      if (isValid) {
        _isEmbeddingsLoaded = true;
        debugPrint("✅ Embeddings model loaded and validated!");
        setState(() => _status = "All models loaded successfully.");
      } else {
        debugPrint("❌ Embeddings model validation failed");
        setState(() => _status = "Embeddings model validation failed.");
      }
    } catch (e) {
      debugPrint("❌ Error loading models: $e");
      setState(() => _status = "Error loading models: $e");
    }
  }

  // ---------------------- Helper for GPT-2 ----------------------

  // Replace your generateAnswer function with this fixed version:

Future<String> generateAnswer(String prompt, {int maxGenLen = 32}) async {
  if (!_isModelLoaded) throw Exception("Model not loaded yet.");

  List<int> tokens = enc.encode(prompt).toList();
  print("Original tokens: ${tokens.take(10).toList()}...");

  // CRITICAL FIX: DistilGPT-2 typically has vocab size 50257, but let's be safe
  const int modelVocabSize = 50257; // Standard GPT-2 vocab size
  const int eosToken = 50256;
  
  // Validate and clamp initial tokens
  tokens = tokens.map((token) {
    if (token < 0 || token >= modelVocabSize) {
      print("Warning: Invalid token $token, replacing with 0 (pad)");
      return 0; // Use pad token as fallback
    }
    return token;
  }).toList();

  print("Validated tokens: ${tokens.take(10).toList()}...");

  // Limit initial prompt length to avoid memory issues
  if (tokens.length > 1000) {
    tokens = tokens.sublist(tokens.length - 1000); // Keep last 1000 tokens
    print("Truncated to last ${tokens.length} tokens");
  }

  for (int step = 0; step < maxGenLen; step++) {
    try {
      int curLen = tokens.length;
      
      // Prepare input - no need for separate padding, use tokens as-is
      var input = [tokens];

      // Resize to current length dynamically
      _interpreter.resizeInputTensor(0, [1, curLen]);
      _interpreter.allocateTensors();

      // Output [1, curLen, vocabSize]
      var output = List.generate(
          1,
          (_) => List.generate(
              curLen, (_) => List<double>.filled(modelVocabSize, 0.0)));

      _interpreter.run(input, output);

      // Get logits for the last token only
      final lastLogits = output[0][curLen - 1];

      // Apply temperature and pick next token
      int nextId = _sampleToken(lastLogits, temperature: 0.8);
      
      // Validate the predicted token
      if (nextId < 0 || nextId >= modelVocabSize) {
        print("Warning: Model predicted invalid token $nextId, stopping generation");
        break;
      }

      tokens.add(nextId);
      
      if (nextId == eosToken) {
        print("EOS token encountered, stopping generation");
        break;
      }

      // Safety check: prevent runaway generation
      if (tokens.length > 200) {
        print("Maximum token length reached, stopping generation");
        break;
      }

    } catch (e) {
      print("Error during generation step $step: $e");
      break;
    }
  }

  try {
    final result = enc.decode(tokens);
    print("Generated text length: ${result.length}");
    return result;
  } catch (e) {
    print("Error decoding tokens: $e");
    return "Error generating response: $e";
  }
}

/// Improved token sampling with temperature
int _sampleToken(List<double> logits, {double temperature = 1.0}) {
  if (temperature <= 0.0) {
    return _argmax(logits);
  }

  // Apply temperature
  final scaledLogits = logits.map((x) => x / temperature).toList();
  
  // Find max for numerical stability
  final maxLogit = scaledLogits.reduce((a, b) => a > b ? a : b);
  
  // Compute softmax probabilities
  final expLogits = scaledLogits.map((x) => exp(x - maxLogit)).toList();
  final sumExp = expLogits.reduce((a, b) => a + b);
  final probs = expLogits.map((x) => x / sumExp).toList();
  
  // Sample from top-k tokens for better quality
  final topK = 50;
  final indexedProbs = probs
      .asMap()
      .entries
      .map((e) => {'index': e.key, 'prob': e.value})
      .toList();

  indexedProbs.sort((a, b) => b['prob']?.compareTo(a['prob'] ?? 0) ?? 0);
  final topKProbs = indexedProbs.take(topK).toList();
  
  // Simple random selection from top-K (you might want a proper random generator)
  final rng = Random();
  final r = rng.nextDouble();
  double cumProb = 0;
  for (var t in topKProbs) {
    cumProb += t['prob'] as double;
    if (r < cumProb) return t['index'] as int;
  }
  return topKProbs.first['index'] as int; // fallback
  }

/// Helper to pick max index (fallback for temperature=0)
int _argmax(List<double> logits) {
  int maxIndex = 0;
  double maxVal = logits[0];
  for (int i = 1; i < logits.length; i++) {
    if (logits[i] > maxVal) {
      maxVal = logits[i];
      maxIndex = i;
    }
  }
  return maxIndex;
}

// Add this import to your main file:



  // ---------------------------------------------------------------

  /// Always work inside /enc_files
  Future<String> _appDirPath() async {
    final dir = await getApplicationDocumentsDirectory();
    final encDir = Directory("${dir.path}/enc_files");
    if (!await encDir.exists()) {
      await encDir.create(recursive: true);
    }
    return encDir.path;
  }

  /// Create a test .txt file with sample content
  Future<void> _createTestFile() async {
    final dirPath = await _appDirPath();
    final timestamp = DateTime.now().millisecondsSinceEpoch;
    final file = File("$dirPath/sample_$timestamp.txt");

    final sampleText = """
This is a sample document about artificial intelligence and machine learning.
AI has revolutionized many industries including healthcare, finance, and technology.
Machine learning algorithms can process large amounts of data to find patterns.
Natural language processing helps computers understand human language.
Deep learning uses neural networks to solve complex problems.
The future of AI looks very promising with many exciting developments ahead.
""";

    await file.writeAsString(sampleText, flush: true);
    setState(() => _status = "Created test file at ${file.path}");
  }

  /// Encrypt all .txt files in /enc_files → .enc
  Future<void> _encryptAll() async {
    setState(() => _status = "Encrypting files...");
    final dirPath = await _appDirPath();
    final dir = Directory(dirPath);

    final files = dir
        .listSync()
        .where((e) => e is File && e.path.endsWith(".txt"))
        .cast<File>()
        .toList();

    if (files.isEmpty) {
      setState(() => _status = "No .txt files found in $dirPath");
      return;
    }

    final crypto = CryptoService();
    await crypto.loadKeyEncrypt();

    int successCount = 0;
    for (final f in files) {
      try {
        final outPath = f.path.replaceFirst(RegExp(r'\.txt$'), '.enc');
        await crypto.encryptFile(f, outPath);
        await f.delete();
        successCount++;
      } catch (e, st) {
        print("❌ Failed to encrypt ${f.path}: $e\n$st");
      }
    }

    setState(() => _status = "Encrypted $successCount/${files.length} files");
  }

  /// Decrypt all .enc files in /enc_files → restore original .txt
  Future<void> _decryptAll() async {
    setState(() => _status = "Decrypting files...");
    final dirPath = await _appDirPath();
    final dir = Directory(dirPath);

    final files = dir
        .listSync()
        .where((e) => e is File && e.path.endsWith(".enc"))
        .cast<File>()
        .toList();

    if (files.isEmpty) {
      setState(() => _status = "No .enc files found in $dirPath");
      return;
    }

    final crypto = CryptoService();
    await crypto.loadKeyDecrypt();

    int successCount = 0;
    for (final f in files) {
      try {
        await crypto.decryptFile(f); // restores .txt + deletes .enc
        successCount++;
      } catch (e, st) {
        print("❌ Failed to decrypt ${f.path}: $e\n$st");
      }
    }

    setState(() => _status = "Decrypted $successCount/${files.length} files");
  }

  /// Run pipeline (decrypt → chunk → embeddings → vector store → re-encrypt)
  Future<void> _runPipeline() async {
    setState(() => _status = "Running pipeline...");
    try {
      final pipeline = Pipeline(
        aesKey32: List.filled(32, 1),
      ); // dummy key for now
      final result = await pipeline.run();
      setState(
        () => _status =
            "Pipeline done: ${result.totalChunks} chunks, ${result.totalEmbeddings} embeddings, refreshed=${result.refreshed}",
      );
    } catch (e) {
      setState(() => _status = "Pipeline failed: $e");
    }
  }

  /// Ask a question against the embeddings in VectorStore
  Future<void> _askQuestion() async {
    final query = _chatController.text.trim();
    if (query.isEmpty) return;

    // Check if models are loaded
    if (!_isModelLoaded || !_isEmbeddingsLoaded || _cachedEmbeddings == null) {
      setState(() {
        _chatResponse = "Models are still loading. Please wait...";
        _status = "Models not ready yet.";
      });
      return;
    }

    PerformanceMonitor.startTimer("Full Question Processing");
    PerformanceMonitor.logMemoryUsage("Before question processing");

    setState(() {
      _status = "Searching embeddings...";
      _chatResponse = "Searching for relevant information...";
    });

    try {
      // Use cached embeddings model for better performance
      PerformanceMonitor.startTimer("Open Vector Store");
      final store = await VectorStore.open(embedSize: 384);
      PerformanceMonitor.endTimer("Open Vector Store");
      
      setState(() => _status = "Generating query embedding...");
      PerformanceMonitor.startTimer("Generate Query Embedding");
      final queryVec = _cachedEmbeddings!.embedTexts([query])[0];
      PerformanceMonitor.endTimer("Generate Query Embedding");
      
      // Check if embedding is valid (not all zeros)
      final isValidEmbedding = queryVec.any((val) => val.abs() > 1e-6);
      if (!isValidEmbedding) {
        setState(() {
          _chatResponse = "Failed to generate valid embedding for your query. Please try a different question.";
          _status = "Embedding generation failed.";
        });
        await store.close();
        PerformanceMonitor.endTimer("Full Question Processing");
        return;
      }
      
      setState(() => _status = "Searching vector database...");
      PerformanceMonitor.startTimer("Vector Search");
      final results = await store.search(queryVec, topK: 3);
      PerformanceMonitor.endTimer("Vector Search");
      await store.close();

      if (results.isEmpty) {
        setState(() {
          _chatResponse = "I couldn't find relevant information for your question in my knowledge base. Could you try rephrasing your question?";
          _status = "No relevant results found.";
        });
        PerformanceMonitor.endTimer("Full Question Processing");
        return;
      }

      final cleanedContext = results
          .map((r) => r.text
              .replaceAll(RegExp(r'[^a-zA-Z0-9\s\.,!?]'), '')
              .replaceAll(RegExp(r'\s+'), ' ')
              .trim())
          .join("\n\n")
          .trim();

      if (cleanedContext.isEmpty) {
        setState(() {
          _chatResponse = "The context doesn't contain the answer to your question.";
          _status = "Context is empty.";
        });
        PerformanceMonitor.endTimer("Full Question Processing");
        return;
      }

      setState(() => _status = "Generating answer...");

      final ragPrompt = """
Please answer only using the context if missing say "The context doesn't contain the answer to your question."
Context: $cleanedContext
Question: $query
Answer:
""";

      PerformanceMonitor.startTimer("Generate Answer");
      final answer = await generateAnswer(ragPrompt, maxGenLen: 64);
      PerformanceMonitor.endTimer("Generate Answer");
      
      String cleanedAnswer = answer.trim();

      // Clean up the response
      if (cleanedAnswer.contains("Answer:")) {
        final parts = cleanedAnswer.split("Answer:");
        cleanedAnswer = parts.length > 1 ? parts.last.trim() : cleanedAnswer;
      }

      if (cleanedAnswer.contains("Context:")) {
        cleanedAnswer = cleanedAnswer.split("Context:").first.trim();
      }

      if (cleanedAnswer.contains("Question:")) {
        cleanedAnswer = cleanedAnswer.split("Question:").first.trim();
      }

      cleanedAnswer = cleanedAnswer
          .replaceAll(RegExp(r'^(Please answer|Context:|Question:).*', multiLine: true), '')
          .trim();

      PerformanceMonitor.endTimer("Full Question Processing");
      PerformanceMonitor.logMemoryUsage("After question processing");
      PerformanceMonitor.printSummary();

      setState(() {
        _chatResponse = cleanedAnswer.isEmpty
            ? "I couldn't generate a proper response."
            : cleanedAnswer;
        _status = "Ready";
      });
    } catch (e) {
      PerformanceMonitor.endTimer("Full Question Processing");
      setState(() {
        _chatResponse = "Error during search: $e";
        _status = "Error occurred";
      });
    }
  }

  /// Debug: list files in /enc_files
  Future<void> _listFiles() async {
    final dirPath = await _appDirPath();
    final dir = Directory(dirPath);
    final files = dir.listSync();
    for (final f in files) {
      print(" - ${f.path}");
    }
    setState(() => _status = "Listed ${files.length} files (see console)");
  }
  
  /// Debug embeddings generation
  Future<void> _debugEmbeddings() async {
    if (_cachedEmbeddings == null) {
      setState(() => _status = "Embeddings model not loaded");
      return;
    }
    
    setState(() => _status = "Running embedding diagnostics...");
    
    try {
      await DebugUtils.testEmbeddingGeneration(_cachedEmbeddings!);
      setState(() => _status = "Embedding diagnostics completed (check console)");
    } catch (e) {
      setState(() => _status = "Embedding diagnostics failed: $e");
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("File Encrypt/Decrypt + Embeddings")),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: SingleChildScrollView(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              ElevatedButton.icon(
                icon: const Icon(Icons.note_add),
                label: const Text("Create Test File"),
                onPressed: _createTestFile,
              ),
              const SizedBox(height: 12),
              ElevatedButton.icon(
                icon: const Icon(Icons.lock),
                label: const Text("Encrypt All .txt Files"),
                onPressed: _encryptAll,
              ),
              const SizedBox(height: 12),
              ElevatedButton.icon(
                icon: const Icon(Icons.lock_open),
                label: const Text("Decrypt All .enc Files"),
                onPressed: _decryptAll,
              ),
              const SizedBox(height: 12),
              ElevatedButton.icon(
                icon: const Icon(Icons.play_circle_fill),
                label: const Text("Run Pipeline"),
                onPressed: _runPipeline,
              ),
              const SizedBox(height: 24),
              TextField(
                controller: _chatController,
                decoration: const InputDecoration(
                  border: OutlineInputBorder(),
                  labelText: "Ask a question",
                  hintText: "e.g., 'What is artificial intelligence?'",
                ),
              ),
              const SizedBox(height: 12),
              ElevatedButton.icon(
                icon: const Icon(Icons.question_answer),
                label: Text((_isModelLoaded && _isEmbeddingsLoaded) ? "Ask" : "Loading Models..."),
                onPressed: (_isModelLoaded && _isEmbeddingsLoaded) ? _askQuestion : null,
              ),
              const SizedBox(height: 12),
              if (_chatResponse.isNotEmpty)
                Container(
                  padding: const EdgeInsets.all(12),
                  decoration: BoxDecoration(
                    border: Border.all(color: Colors.grey),
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: Text(
                    _chatResponse,
                    style: const TextStyle(fontSize: 14),
                  ),
                ),
              const SizedBox(height: 12),
              ElevatedButton.icon(
                icon: const Icon(Icons.list),
                label: const Text("List Files in /enc_files"),
                onPressed: _listFiles,
              ),
              const SizedBox(height: 12),
              ElevatedButton.icon(
                icon: const Icon(Icons.bug_report),
                label: const Text("Debug Embeddings"),
                onPressed: _isEmbeddingsLoaded ? _debugEmbeddings : null,
              ),
              const SizedBox(height: 30),
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: Colors.blue.shade50,
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Text(
                  _status,
                  style: const TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.bold,
                  ),
                  textAlign: TextAlign.center,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}