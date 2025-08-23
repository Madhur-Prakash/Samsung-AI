import 'dart:io';
import 'package:path_provider/path_provider.dart';
import 'crypto_service.dart';
import 'embeddings.dart';
import 'text_chunker.dart';
import 'vector_store.dart';

class PipelineResult {
  final int totalChunks;
  final bool refreshed;
  final String message;
  PipelineResult(this.totalChunks, this.refreshed, this.message);
}

class Pipeline {
  final String folderName;
  final Duration maxAge;
  final List<int> aesKey32;

  Pipeline({
    this.folderName = "enc_files",
    this.maxAge = const Duration(days: 30),
    required this.aesKey32,
  });

  Future<PipelineResult> run() async {
    final baseDir = await getApplicationDocumentsDirectory();
    final textDir = Directory('${baseDir.path}/$folderName');

    if (!await textDir.exists()) {
      await textDir.create(recursive: true);
      print("Directory created at: ${textDir.path}");
    }

    // Decrypt .enc → .txt
    final crypto = CryptoService();
    await crypto.loadKeyDecrypt(); // Load key for decryption, if exists
    final decCount = await crypto.decryptDirectory(textDir.path);
    print("Decrypted $decCount files in: ${textDir.path}");

    final lines = await _readTxtFiles(textDir.path);
    print("Collected ${lines.length} lines from text files");

    if (lines.isEmpty) {
      return PipelineResult(
        0,
        false,
        decCount == 0
            ? "No encrypted or text files found."
            : "Decrypted files, but no valid lines.",
      );
    }

    // Chunking
    final chunker = TextChunker(chunkSize: 500, overlap: 100);
    final chunks = chunker.chunkLines(lines);
    print("Created ${chunks.length} text chunks");

    if (chunks.isEmpty) return PipelineResult(0, false, "No chunks produced.");

    // Embeddings
    final embeddings = await Embeddings.load(
      modelAsset: 'assets/models/sentence_transformer.tflite',
      vocabAsset: 'assets/tokenizer/vocab.txt',
      maxLen: 128,
      embedSize: 384,
    );

    final store = await VectorStore.open(embedSize: 384);

    bool refreshed = false;
    final created = await store.createdTime();
    if (created != null) {
      final age = DateTime.now().difference(created);
      if (age >= maxAge) {
        await store.clear();
        refreshed = true;
        print("Vector store refreshed after $age");
      }
    }

    final items = <({String text, List<double> vec})>[];
    for (final c in chunks) {
      final vec = embeddings.embed(c);
      items.add((text: c, vec: vec));
    }

    await store.upsertBatch(items);
    print("Upserted ${items.length} embeddings into vector store");

    await _resetDir(textDir.path);
    print("Reset directory: ${textDir.path}");

    embeddings.close();
    return PipelineResult(chunks.length, refreshed, "Embeddings stored successfully.");
  }

  Future<List<String>> _readTxtFiles(String dirPath) async {
    final dir = Directory(dirPath);
    if (!await dir.exists()) return [];
    final files = await dir
        .list()
        .where((e) => e is File && e.path.endsWith('.txt'))
        .cast<File>()
        .toList();

    final lines = <String>[];
    for (final f in files) {
      try {
        final content = await f.readAsLines();
        lines.addAll(content);
      } catch (_) {}
    }
    return lines;
  }

  Future<void> _resetDir(String dirPath) async {
    final dir = Directory(dirPath);
    if (await dir.exists()) await dir.delete(recursive: true);
    await Directory(dirPath).create(recursive: true);
  }
}
