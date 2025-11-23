// services/pipeline.dart
import 'dart:io';
import 'package:path_provider/path_provider.dart';
import 'text_chunker.dart';
import 'embeddings.dart';
import 'vector_store.dart';
import 'crypto_service.dart';

class PipelineResult {
  final int totalChunks;
  final int totalEmbeddings;
  final bool refreshed;
  final String message;
  
  PipelineResult(this.totalChunks, this.totalEmbeddings, this.refreshed, this.message);
}

class Pipeline {
  final List<int> aesKey32;
  final TextChunker chunker;
  
  Pipeline({
    required this.aesKey32,
    TextChunker? chunker,
  }) : chunker = chunker ?? TextChunker();

  Future<String> _getEncFilesDir() async {
    final dir = await getApplicationDocumentsDirectory();
    final encDir = Directory("${dir.path}/enc_files");
    if (!await encDir.exists()) {
      await encDir.create(recursive: true);
    }
    return encDir.path;
  }

  Future<PipelineResult> run() async {
    try {
      final dirPath = await _getEncFilesDir();
      final dir = Directory(dirPath);
      
      // Step 1: Decrypt .enc files to .txt (temporarily)
      final encFiles = dir
          .listSync()
          .where((e) => e is File && e.path.endsWith(".enc"))
          .cast<File>()
          .toList();

      if (encFiles.isEmpty) {
        return PipelineResult(0, 0, false, "No .enc files found to process");
      }

      final crypto = CryptoService();
      await crypto.loadKeyDecrypt();

      // Decrypt files temporarily
      List<File> tempTxtFiles = [];
      for (final encFile in encFiles) {
        try {
          await crypto.decryptFile(encFile); // This creates .txt and deletes .enc
          final txtPath = encFile.path.replaceFirst(RegExp(r'\.enc$'), '.txt');
          tempTxtFiles.add(File(txtPath));
        } catch (e) {
          print("❌ Failed to decrypt ${encFile.path}: $e");
        }
      }

      // Step 2: Process text files into chunks
      List<String> allChunks = [];
      List<String> chunkSources = []; // Track which file each chunk came from
      
      for (var file in tempTxtFiles) {
        final content = await file.readAsString();
        final chunks = chunker.chunkText(content);
        allChunks.addAll(chunks);
        
        // Track source file for each chunk
        for (int i = 0; i < chunks.length; i++) {
          chunkSources.add(file.path);
        }
      }

      if (allChunks.isEmpty) {
        return PipelineResult(0, 0, false, "No text content found in decrypted files");
      }

      print("📂 Collected ${allChunks.length} text chunks from ${tempTxtFiles.length} files");

      // Step 3: Generate embeddings with validation
      final embeddings = await Embeddings.load();
      
      // Validate model before processing
      final isValid = await embeddings.validateModel();
      if (!isValid) {
        return PipelineResult(0, 0, false, "Embeddings model validation failed");
      }
      
      print("📊 Generating embeddings for ${allChunks.length} chunks...");
      final vectors = embeddings.embedTexts(allChunks);
      
      // Validate generated embeddings
      int validEmbeddings = 0;
      for (final vector in vectors) {
        final norm = vector.fold(0.0, (sum, val) => sum + val * val);
        if (norm > 1e-6) validEmbeddings++;
      }
      
      print("✅ Generated ${vectors.length} embeddings (${validEmbeddings} valid)");
      
      if (validEmbeddings == 0) {
        return PipelineResult(allChunks.length, 0, false, "No valid embeddings generated");
      }

      // Step 4: Store in vector database with validation
      final vectorStore = await VectorStore.open(embedSize: 384);
      
      // Clear existing data for refresh
      await vectorStore.clear();
      
      // Add only valid embeddings to the store
      int storedCount = 0;
      for (int i = 0; i < allChunks.length; i++) {
        final vector = vectors[i];
        final norm = vector.fold(0.0, (sum, val) => sum + val * val);
        
        // Only store embeddings with meaningful values
        if (norm > 1e-6) {
          await vectorStore.add(
            id: 'chunk_$i',
            embedding: vector,
            text: allChunks[i],
            metadata: {
              'source': chunkSources[i],
              'norm': norm,
              'chunk_index': i,
            },
          );
          storedCount++;
        } else {
          print("Skipping chunk $i with zero embedding");
        }
      }

      await vectorStore.close();
      print("💾 Stored $storedCount/${vectors.length} valid embeddings in vector store");

      // Step 5: Re-encrypt the files
      await crypto.loadKeyEncrypt();
      for (final txtFile in tempTxtFiles) {
        try {
          final encPath = txtFile.path.replaceFirst(RegExp(r'\.txt$'), '.enc');
          await crypto.encryptFile(txtFile, encPath);
          await txtFile.delete(); // Clean up temp file
        } catch (e) {
          print("❌ Failed to re-encrypt ${txtFile.path}: $e");
        }
      }

      return PipelineResult(
        allChunks.length, 
        storedCount, 
        true, 
        "Pipeline completed: $storedCount valid embeddings stored"
      );

    } catch (e, stackTrace) {
      print("❌ Pipeline failed: $e");
      print("Stack trace: $stackTrace");
      return PipelineResult(0, 0, false, "Pipeline failed: $e");
    }
  }
}