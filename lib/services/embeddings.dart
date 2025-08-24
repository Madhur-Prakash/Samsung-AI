import 'package:tflite_flutter/tflite_flutter.dart';
import 'tokenizer.dart';

class Embeddings {
  final Interpreter interpreter;
  final BertTokenizer tokenizer;
  final int maxLen;
  final int embedSize;

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
    print("🔍 Model input details:");
    for (int i = 0; i < interpreter.getInputTensors().length; i++) {
      final tensor = interpreter.getInputTensor(i);
      print("  Input $i: ${tensor.shape}, type: ${tensor.type}");
    }
    
    print("🔍 Model output details:");
    for (int i = 0; i < interpreter.getOutputTensors().length; i++) {
      final tensor = interpreter.getOutputTensor(i);
      print("  Output $i: ${tensor.shape}, type: ${tensor.type}");
    }
    
    return Embeddings._(interpreter, tokenizer, maxLen, embedSize);
  }

  /// Embed multiple texts at once with better error handling
  List<List<double>> embedTexts(List<String> texts) {
    try {
      // Process texts in smaller batches to avoid memory issues
      const batchSize = 4; // Reduced batch size
      List<List<double>> allEmbeddings = [];
      
      for (int start = 0; start < texts.length; start += batchSize) {
        final end = (start + batchSize < texts.length) ? start + batchSize : texts.length;
        final batch = texts.sublist(start, end);
        
        print("🔄 Processing batch ${start ~/ batchSize + 1}/${(texts.length / batchSize).ceil()}: ${batch.length} texts");
        
        final batchEmbeddings = _embedBatch(batch);
        allEmbeddings.addAll(batchEmbeddings);
      }
      
      return allEmbeddings;
    } catch (e, stackTrace) {
      print("❌ Error in embedTexts: $e");
      print("Stack trace: $stackTrace");
      rethrow;
    }
  }

  List<List<double>> _embedBatch(List<String> texts) {
    // Prepare inputs with better validation
    final inputs = <List<int>>[];
    
    for (int i = 0; i < texts.length; i++) {
      final text = texts[i];
      print("🔤 Processing text $i: '${text.length > 50 ? text.substring(0, 50) + '...' : text}'");
      
      try {
        final tokens = tokenizer.encode(text);
        print("🔢 Tokens (first 10): ${tokens.take(10).toList()}...");
        
        // Validate token IDs - they should be within vocabulary range
        final invalidTokens = tokens.where((token) => token < 0 || token >= tokenizer.vocabSize);
        if (invalidTokens.isNotEmpty) {
          print("⚠️ Warning: Found invalid tokens: ${invalidTokens.toList()}");
          // Replace invalid tokens with UNK token (usually ID 1 or 100)
          final cleanedTokens = tokens.map((t) => (t < 0 || t >= tokenizer.vocabSize) ? 1 : t).toList();
          tokens.clear();
          tokens.addAll(cleanedTokens);
        }
        
        // Add special tokens if not already present
        List<int> processedTokens = [];
        
        // Add CLS token at beginning if not present (usually ID 101)
        if (tokens.isEmpty || tokens.first != 101) {
          processedTokens.add(101); // CLS token
        }
        
        // Add the text tokens (but leave room for SEP token)
        final availableLength = maxLen - 2; // Reserve space for CLS and SEP
        if (tokens.length > availableLength) {
          processedTokens.addAll(tokens.sublist(0, availableLength));
        } else {
          processedTokens.addAll(tokens);
        }
        
        // Add SEP token at end (usually ID 102)
        if (processedTokens.length < maxLen) {
          processedTokens.add(102); // SEP token
        }
        
        // Pad or truncate to maxLen
        final padded = List<int>.filled(maxLen, 0); // PAD token is usually 0
        for (int j = 0; j < maxLen && j < processedTokens.length; j++) {
          padded[j] = processedTokens[j];
        }
        
        print("🎯 Final tokens (first 10): ${padded.take(10).toList()}...");
        inputs.add(padded);
      } catch (e) {
        print("❌ Error tokenizing text $i: $e");
        // Add a safe default input
        final safeInput = List<int>.filled(maxLen, 0);
        safeInput[0] = 101; // CLS
        safeInput[1] = 102; // SEP
        inputs.add(safeInput);
      }
    }

    try {
      // Prepare the input tensor
      final inputTensor = inputs;
      print("📊 Input tensor shape: [${inputTensor.length}, ${inputTensor[0].length}]");
      
      // Prepare the output tensor
      final output = List.generate(
        texts.length, 
        (i) => List<double>.filled(embedSize, 0.0)
      );
      
      print("📤 Output tensor shape: [${output.length}, ${output[0].length}]");
      
      // Run inference
      print("🚀 Running inference...");
      interpreter.run(inputTensor, output);
      print("✅ Inference completed successfully");
      
      return List<List<double>>.from(output.map((e) => e.cast<double>()));
      
    } catch (e, stackTrace) {
      print("❌ TensorFlow Lite inference failed: $e");
      print("Stack trace: $stackTrace");
      
      // Return zero embeddings as fallback
      print("🔄 Returning zero embeddings as fallback");
      return List.generate(texts.length, (i) => List<double>.filled(embedSize, 0.0));
    }
  }
}