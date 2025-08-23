class TextChunker {
  final int chunkSize;
  final int overlap;

  TextChunker({this.chunkSize = 500, this.overlap = 100});

  List<String> chunkLines(List<String> lines) {
    final filtered = lines.where((l) => l.trim().length > 10).toList();
    final List<String> chunks = [];
    final sb = StringBuffer();

    for (final line in filtered) {
      final candidate = (sb.isEmpty ? '' : '${sb.toString()} ') + line.trim();
      if (candidate.length > chunkSize) {
        final current = sb.toString().trim();
        if (current.isNotEmpty) chunks.add('passage: $current');

        final tail = current.length > overlap
            ? current.substring(current.length - overlap)
            : current;
        sb.clear();
        sb.write('$tail ${line.trim()}');
      } else {
        sb.clear();
        sb.write(candidate);
      }
    }

    final tail = sb.toString().trim();
    if (tail.isNotEmpty) chunks.add('passage: $tail');
    return chunks;
  }
}
