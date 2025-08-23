import 'dart:math' as math;
import 'dart:typed_data';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';
import 'package:sqflite/sqflite.dart';
import 'dart:io';

class VectorStore {
  final Database db;
  final int embedSize;

  VectorStore._(this.db, this.embedSize);

  static Future<String> _dbPath() async {
    final dir = await getApplicationDocumentsDirectory();
    return p.join(dir.path, 'img_vector_store.sqlite');
  }

  static Future<VectorStore> open({int embedSize = 384}) async {
    final path = await _dbPath();
    final db = await openDatabase(
      path,
      version: 1,
      onCreate: (d, v) async {
        await d.execute('''
          CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            vec BLOB NOT NULL
          );
        ''');
      },
    );
    return VectorStore._(db, embedSize);
  }

  Future<void> clear() async {
    await db.delete('embeddings');
  }

  Future<void> upsertBatch(List<({String text, List<double> vec})> items) async {
    final batch = db.batch();
    for (final item in items) {
      batch.insert('embeddings', {
        'text': item.text,
        'vec': _floatsToBytes(item.vec),
      });
    }
    await batch.commit(noResult: true);
  }

  Future<List<Map<String, dynamic>>> all() async {
    return db.query('embeddings');
  }

  Future<DateTime?> createdTime() async {
    final path = await _dbPath();
    try {
      final fileStat = await File(path).stat();
      return fileStat.changed;
    } catch (_) {
      return null;
    }
  }

  Future<List<({String text, double score})>> search(List<double> queryVec, {int topK = 5}) async {
    final rows = await all();
    final results = <({String text, double score})>[];

    for (final r in rows) {
      final blob = r['vec'] as Uint8List;
      final text = r['text'] as String;
      final vec = _bytesToFloats(blob);
      final score = _cosine(vec, queryVec);
      results.add((text: text, score: score));
    }

    results.sort((a, b) => b.score.compareTo(a.score));
    if (results.length > topK) return results.take(topK).toList();
    return results;
  }

  // --- Helpers ---
  Uint8List _floatsToBytes(List<double> floats) {
    final b = ByteData(4 * floats.length);
    for (int i = 0; i < floats.length; i++) {
      b.setFloat32(i * 4, floats[i], Endian.little);
    }
    return b.buffer.asUint8List();
  }

  List<double> _bytesToFloats(Uint8List bytes) {
    final b = ByteData.sublistView(bytes);
    final len = bytes.length ~/ 4;
    final out = List<double>.filled(len, 0);
    for (int i = 0; i < len; i++) {
      out[i] = b.getFloat32(i * 4, Endian.little);
    }
    return out;
  }

  double _cosine(List<double> a, List<double> b) {
    double dot = 0, na = 0, nb = 0;
    for (int i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
      na += a[i] * a[i];
      nb += b[i] * b[i];
    }
    return dot / (math.sqrt(na) * math.sqrt(nb) + 1e-9);
  }
}
