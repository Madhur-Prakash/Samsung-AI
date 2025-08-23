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
    int embedSize = 384, // depends on your model
  }) async {
    final interpreter = await Interpreter.fromAsset(modelAsset);
    final tokenizer = await BertTokenizer.fromAsset(vocabAsset);
    return Embeddings._(interpreter, tokenizer, maxLen, embedSize);
  }

  /// Returns a float list of size [embedSize].
  List<double> embed(String text) {
    final ids = tokenizer.encode(text, maxLen: maxLen);

    // Model-dependent: often BERT style expects input_ids, attention_mask, token_type_ids.
    // If your exported model expects only input_ids, simplify accordingly.
    final inputIds = List<List<int>>.from([ids]);
    final attention = List<List<int>>.from([
      ids.map((e) => e == 0 ? 0 : 1).toList()  // crude mask; adjust if needed
    ]);
    final tokenTypes = List<List<int>>.from([List.filled(maxLen, 0)]);

    // Prepare output
    final output = List.filled(embedSize, 0.0).reshape([1, embedSize]);

    // Try run with 1 or 3 inputs depending on the model signature.
    try {
      interpreter.run({'input_ids': inputIds, 'attention_mask': attention, 'token_type_ids': tokenTypes}, {'embeddings': output});
    } catch (_) {
      // Fallback: common case with a single input and single output.
      interpreter.run(inputIds, output);
    }

    return List<double>.from(output[0]);
  }

  void close() => interpreter.close();
}
