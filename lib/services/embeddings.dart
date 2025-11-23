import 'dart:math';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'tokenizer.dart';

class Embeddings {
  final Interpreter interpreter;
  final BertTokenizer tokenizer;
  final int maxLen;
  final int embedSize;
  late int _modelVocabSize;

  Embeddings._(this.interpreter, this.tokenizer, this.maxLen, this.embedSize);

  static Future<Embeddings> load({
    String modelAsset = 'assets/models/sentence_transformer.tflite',
    String vocabAsset = 'assets/tokenizer/vocab.txt',
    int maxLen = 128, 
    int embedSize = 384,
  }) async {
    final interpreter = await Interpreter.fromAsset(modelAsset);
    final tokenizer = await BertTokenizer.fromAsset(vocabAsset);
    
    // Print model info for debugging
    print("Model input details:");
    for (int i = 0; i < interpreter.getInputTensors().length; i++) {
      final tensor = interpreter.getInputTensor(i);
      print("  Input $i: ${tensor.shape}, type: ${tensor.type}");
    }
    
    print("Model output details:");
    for (int i = 0; i < interpreter.getOutputTensors().length; i++) {
      final tensor = interpreter.getOutputTensor(i);
      print("  Output $i: ${tensor.shape}, type: ${tensor.type}");
    }
    
    final embeddings = Embeddings._(interpreter, tokenizer, maxLen, embedSize);
    
    // all-MiniLM-L6-v2 uses the same vocab as BERT but we need to be more careful
    // The model actually uses vocab size 30522 (standard BERT vocab)
    embeddings._modelVocabSize = 30522;
    print("all-MiniLM-L6-v2 model vocabulary size: ${embeddings._modelVocabSize}");
    
    return embeddings;
  }

  List<List<double>> embedTexts(List<String> texts) {
    if (texts.isEmpty) {
      print("No texts provided for embedding");
      return [];
    }
    
    try {
      List<List<double>> allEmbeddings = [];
      
      for (int i = 0; i < texts.length; i++) {
        final text = texts[i].trim();
        if (text.isEmpty) {
          print("Skipping empty text at index $i");
          allEmbeddings.add(List<double>.filled(embedSize, 0.0));
          continue;
        }
        
        final embedding = _embedSingle(text);
        allEmbeddings.add(embedding);
        
        // Add small delay to prevent overwhelming the model
        if (i < texts.length - 1) {
          Future.delayed(const Duration(milliseconds: 10));
        }
      }
      
      print("Successfully generated ${allEmbeddings.length} embeddings");
      return allEmbeddings;
    } catch (e, stackTrace) {
      print("Error in embedTexts: $e");
      print("Stack trace: $stackTrace");
      rethrow;
    }
  }

  List<double> _embedSingle(String text) {
    try {
      if (text.trim().isEmpty) {
        print("Empty text provided, returning zero embedding");
        return List<double>.filled(embedSize, 0.0);
      }

      // Tokenize the text
      final tokens = tokenizer.encode(text, maxLen: maxLen);
      
      // Enhanced token validation for all-MiniLM-L6-v2
      final validTokens = tokens.map((token) {
        // Clamp tokens to valid range
        if (token < 0) return 0; // PAD token
        if (token >= _modelVocabSize) return 100; // UNK token
        return token;
      }).toList();

      // Create attention mask (1 for real tokens, 0 for padding)
      final attentionMask = validTokens.map((id) => id != 0 ? 1 : 0).toList();
      
      // Ensure we have at least some non-padding tokens
      final nonPadTokens = attentionMask.where((mask) => mask == 1).length;
      if (nonPadTokens == 0) {
        print("No valid tokens found, returning zero embedding");
        return List<double>.filled(embedSize, 0.0);
      }

      // Prepare inputs for all-MiniLM-L6-v2
      final inputIds = [validTokens];
      final attentionMasks = [attentionMask];

      // Resize tensors dynamically
      interpreter.resizeInputTensor(0, [1, maxLen]);
      interpreter.resizeInputTensor(1, [1, maxLen]);
      interpreter.allocateTensors();

      // Prepare output buffer
      final output = [List<double>.filled(embedSize, 0.0)];

      // Run inference
      interpreter.runForMultipleInputs([inputIds, attentionMasks], {0: output});
      
      // Validate output embedding
      final embedding = output[0];
      final embeddingNorm = _calculateNorm(embedding);
      
      if (embeddingNorm < 1e-6) {
        print("Warning: Generated embedding has very low norm ($embeddingNorm)");
        // Still return it as it might be valid
      }
      
      return embedding;
    } catch (e, stackTrace) {
      print("TensorFlow Lite inference failed: $e");
      print("Stack trace: $stackTrace");
      
      // Return zero embedding as fallback
      return List<double>.filled(embedSize, 0.0);
    }
  }

  // Helper method to validate model compatibility
  Future<bool> validateModel() async {
    try {
      const testText = "This is a test sentence.";
      final embedding = _embedSingle(testText);
      final hasNonZero = embedding.any((val) => val.abs() > 1e-6);
      final embeddingNorm = _calculateNorm(embedding);
      print("Model validation:");
      print("  - Non-zero values: $hasNonZero");
      print("  - Embedding norm: $embeddingNorm");
      print("  - Sample values: ${embedding.take(5).toList()}");
      return hasNonZero && embeddingNorm > 0.01;
    } catch (e) {
      print("Model validation failed: $e");
      return false;
    }
  }

  double _calculateNorm(List<double> vector) {
    double sum = 0.0;
    for (final val in vector) {
      sum += val * val;
    }
    return sum > 0 ? sqrt(sum) : 0.0;
  }
}