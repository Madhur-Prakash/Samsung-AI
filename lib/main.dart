import 'dart:io';
import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';
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

  /// Always work inside /enc_files
  Future<String> _appDirPath() async {
    final dir = await getApplicationDocumentsDirectory();
    final encDir = Directory("${dir.path}/enc_files");
    print("App documents directory: ${dir.path}");
    print("Working in enc_files directory: ${encDir.path}");
    if (!await encDir.exists()) {
      await encDir.create(recursive: true);
    }
    return encDir.path;
  }

  /// Create a test .txt file with "hello"
  Future<void> _createTestFile() async {
    final dirPath = await _appDirPath();
    final file = File("$dirPath/${DateTime.now().millisecondsSinceEpoch}.txt");
    await file.writeAsString("hello, how u doing?", flush: true);
    print("Created test file at ${file.path}");
    setState(() => _status = "Created ${file.path} with 'hello'");
  }

  /// Encrypt all `.txt` files in /enc_files → `.enc`
 Future<void> _encryptAll() async {
  setState(() => _status = "Encrypting files...");

  final dirPath = await _appDirPath();
  final dir = Directory(dirPath);

  // Collect only .txt files
  final files = dir
      .listSync()
      .where((e) => e is File && e.path.endsWith(".txt"))
      .cast<File>()
      .toList();

  if (files.isEmpty) {
    setState(() => _status = "No .txt files found in $dirPath");
    return;
  }

  print("Encrypting ${files.length} .txt files in $dirPath");

  final crypto = CryptoService();
  await crypto.loadKeyEncrypt();

  int successCount = 0;

  for (final f in files) {
    try {
      final outPath = f.path.replaceFirst(RegExp(r'\.txt$'), '.enc');

      await crypto.encryptFile(f, outPath);

      // Remove original .txt after successful encryption
      await f.delete();

      print("Encrypted: ${f.path} → $outPath");
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

  print("Decrypting ${files.length} .enc files in $dirPath");

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


  /// Run pipeline (decrypt → chunk → embeddings → vector store)
  Future<void> _runPipeline() async {
    setState(() => _status = "Running pipeline...");
    final pipeline =
        Pipeline(aesKey32: List.filled(32, 1)); // dummy key, not used in init()
    final result = await pipeline.run();

    setState(() => _status =
        "Pipeline done: ${result.totalChunks} chunks, refreshed=${result.refreshed}");
  }

  /// Debug: list files in /enc_files
  Future<void> _listFiles() async {
    final dirPath = await _appDirPath();
    final dir = Directory(dirPath);
    final files = dir.listSync();

    print("Files in $dirPath:");
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
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            ElevatedButton.icon(
              icon: const Icon(Icons.note_add),
              label: const Text("Create Test File (hello)"),
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
            const SizedBox(height: 12),
            ElevatedButton.icon(
              icon: const Icon(Icons.list),
              label: const Text("List Files in /enc_files"),
              onPressed: _listFiles,
            ),
            const SizedBox(height: 30),
            Text(
              _status,
              style: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
              textAlign: TextAlign.center,
            ),
          ],
        ),
      ),
    );
  }
}
