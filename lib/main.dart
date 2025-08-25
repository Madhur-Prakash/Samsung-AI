import 'dart:io';
import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:samsung_ai/services/embeddings.dart';
import 'package:samsung_ai/services/vector_store.dart';
import 'services/crypto_service.dart';
import 'services/pipeline.dart';

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
  String _status = "Idle";
  final TextEditingController _chatController = TextEditingController();
  String _chatResponse = "";

  late Interpreter _interpreter;

  @override
  void initState() {
    super.initState();
    loadModel();
  }

  Future<void> loadModel() async {
    _interpreter = await Interpreter.fromAsset('assets/models/distilgpt2.tflite');
  }

  // ---------------------- Helper for GPT-2 ----------------------
  List<int> tokenize(String text) {
    // Simple byte-level tokenizer (mock, works for ASCII)
    return text.codeUnits;
  }

  String detokenize(List<int> ids) {
    // Convert byte IDs back to string
    return String.fromCharCodes(ids.where((id) => id != 0));
  }

  Future<String> generateAnswer(String prompt) async {
  // Tokenize input
  final inputIds = tokenize(prompt);

  // For TFLite, we need fixed input length, e.g., 128
  final inputLength = 128;
  final inputTensor = List.generate(inputLength, (i) => i < inputIds.length ? inputIds[i] : 0);

  // Output shape is [1, 128, 50257] (batch, sequence, vocab)
  final outputTensor = List.generate(
    1,
    (_) => List.generate(
      128,
      (_) => List.filled(50257, 0.0),
    ),
  );

  // Run TFLite model
  _interpreter.run([inputTensor], outputTensor);

  // Extract predicted token IDs using argmax on the vocab dimension
  List<int> predictedIds = [];
  for (int i = 0; i < 128; i++) {
    List<double> logits = outputTensor[0][i];  // Correct: batch 0, position i
    int maxIndex = 0;
    double maxValue = logits[0];
    for (int j = 1; j < logits.length; j++) {
      if (logits[j] > maxValue) {
        maxValue = logits[j];
        maxIndex = j;
      }
    }
    predictedIds.add(maxIndex);
  }


  // Detokenize output
  return detokenize(predictedIds);
}

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

  /// Encrypt all `.txt` files in /enc_files → `.enc`
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

  /// Decrypt all `.enc` files in /enc_files → restore original `.txt`
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
      final pipeline = Pipeline(aesKey32: List.filled(32, 1)); // dummy key for now
      final result = await pipeline.run();
      setState(() => _status =
          "Pipeline done: ${result.totalChunks} chunks, ${result.totalEmbeddings} embeddings, refreshed=${result.refreshed}");
    } catch (e) {
      setState(() => _status = "Pipeline failed: $e");
    }
  }

  /// Ask a question against the embeddings in VectorStore
  Future<void> _askQuestion() async {
    final query = _chatController.text.trim();
    if (query.isEmpty) return;

    setState(() => _status = "Searching embeddings...");

    try {
      final embeddings = await Embeddings.load();
      final store = await VectorStore.open(embedSize: 384);

      // Embed single query
      final queryVec = embeddings.embedTexts([query])[0];

      // Get top K results
      final results = await store.search(queryVec, topK: 3);
      await store.close();

      if (results.isEmpty) {
        setState(() {
          _chatResponse = "No relevant chunks found.";
          _status = "Q&A done.";
        });
      } else {
        // Combine chunks into context string
        final context = results.map((r) => r.text).join("\n\n");

        // Generate formatted answer using DistilGPT-2 TFLite
        try {
          final formattedAnswer = await generateAnswer(context);
          setState(() {
            _chatResponse = formattedAnswer;
            _status = "Q&A done - formatted answer generated.";
          });
        } catch (e) {
          setState(() {
            _chatResponse = "Error generating answer: $e";
            _status = "Q&A failed.";
          });
        }
      }
    } catch (e) {
      setState(() {
        _chatResponse = "Error during search: $e";
        _status = "Q&A failed.";
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
                label: const Text("Ask"),
                onPressed: _askQuestion,
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
              const SizedBox(height: 30),
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: Colors.blue.shade50,
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Text(
                  _status,
                  style: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
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
