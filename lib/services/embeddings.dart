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
    return Embeddings._(interpreter, tokenizer, maxLen, embedSize);
  }

  List<double> embed(String text) {
    final ids = tokenizer.encode(text, maxLen: maxLen);
    final inputIds = List<List<int>>.from([ids]);
    final attention = List<List<int>>.from([
      ids.map((e) => e == 0 ? 0 : 1).toList()
    ]);
    final tokenTypes = List<List<int>>.from([List.filled(maxLen, 0)]);
    final output = List.filled(embedSize, 0.0).reshape([1, embedSize]);

    try {
      interpreter.run({'input_ids': inputIds, 'attention_mask': attention, 'token_type_ids': tokenTypes}, {'embeddings': output});
    } catch (_) {
      interpreter.run(inputIds, output);
    }

    return List<double>.from(output[0]);
  }

  void close() => interpreter.close();
}
