import 'dart:convert';
import 'package:flutter/services.dart' show rootBundle;

class BertTokenizer {
  late final Map<String, int> vocab;
  final String unkToken;
  final bool doLowerCase;

  BertTokenizer._(this.vocab, {this.unkToken = "[UNK]", this.doLowerCase = true});

  static Future<BertTokenizer> fromAsset(String assetPath,
      {String unkToken = "[UNK]", bool doLowerCase = true}) async {
    final vocabText = await rootBundle.loadString(assetPath);
    final lines = const LineSplitter().convert(vocabText);
    final map = <String, int>{};
    for (var i = 0; i < lines.length; i++) map[lines[i].trim()] = i;
    return BertTokenizer._(map, unkToken: unkToken, doLowerCase: doLowerCase);
  }

  List<int> encode(String text, {int maxLen = 128}) {
    final tokens = _tokenize(text);
    final wp = ["[CLS]", ...tokens, "[SEP]"];
    final ids = wp.map((t) => vocab[t] ?? vocab[unkToken]!).toList();
    final padded = List<int>.filled(maxLen, vocab["[PAD]"] ?? 0);
    for (int i = 0; i < ids.length && i < maxLen; i++) padded[i] = ids[i];
    return padded;
  }

  List<String> _tokenize(String text) {
    String t = doLowerCase ? text.toLowerCase() : text;
    t = t.replaceAll(RegExp(r'\s+'), ' ').trim();
    final words = t.split(RegExp(r"\s+")).where((w) => w.isNotEmpty).toList();
    final List<String> output = [];
    for (final w in words) output.addAll(_wordpiece(w));
    return output;
  }

  List<String> _wordpiece(String token) {
    if (vocab.containsKey(token)) return [token];
    final List<String> subTokens = [];
    int start = 0;
    while (start < token.length) {
      int end = token.length;
      String curSubStr = "";
      while (start < end) {
        var substr = token.substring(start, end);
        if (start > 0) substr = "##$substr";
        if (vocab.containsKey(substr)) {
          curSubStr = substr;
          break;
        }
        end -= 1;
      }
      if (curSubStr.isEmpty) return [unkToken];
      subTokens.add(curSubStr);
      start = end;
    }
    return subTokens;
  }
}
