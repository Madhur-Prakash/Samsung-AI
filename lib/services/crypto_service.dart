import 'dart:io';
import 'dart:typed_data';
import 'package:encrypt/encrypt.dart' as enc;
import 'package:path/path.dart' as p;

class CryptoService {
  // Example AES-256-GCM. You must provide the real key/iv and how .enc is structured.
  final Uint8List keyBytes; // 32 bytes
  CryptoService(this.keyBytes);

  Future<int> decryptDirectory(String dirPath) async {
    final dir = Directory(dirPath);
    if (!await dir.exists()) return 0;

    final files = await dir
        .list()
        .where((e) => e is File && e.path.endsWith('.enc'))
        .cast<File>()
        .toList();

    int count = 0;
    for (final f in files) {
      try {
        final data = await f.readAsBytes();

        // Example simple layout: [12-byte nonce][ciphertext][16-byte tag]
        if (data.length < 12 + 16) continue;
        final nonce = data.sublist(0, 12);
        final tag = data.sublist(data.length - 16);
        final cipherText = data.sublist(12, data.length - 16);

        final key = enc.Key(keyBytes);
        final iv = enc.IV(nonce);
        final encr = enc.Encrypter(enc.AES(key, mode: enc.AESMode.gcm));

        // ✅ Fix: convert to Uint8List before passing
        final Uint8List combined = Uint8List.fromList(cipherText + tag);

        final decrypted = encr.decryptBytes(
          enc.Encrypted(combined),
          iv: iv,
        );

        final outPath = p.setExtension(f.path, '.txt');
        await File(outPath).writeAsBytes(decrypted);
        count++;
      } catch (e) {
        // log if needed
        print("Decryption failed for ${f.path}: $e");
      }
    }
    return count;
  }
}
