import 'package:flutter/material.dart';
import 'services/pipeline.dart';
import 'services/embeddings.dart';
import 'services/vector_store.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'On-device RAG',
      theme: ThemeData(useMaterial3: true),
      home: const HomePage(),
      debugShowCheckedModeBanner: false,
    );
  }
}

class HomePage extends StatefulWidget {
  const HomePage({super.key});
  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  String status = 'Idle';
  final queryCtrl = TextEditingController();
  List<MapEntry<String, double>> results = [];

  // ⚠️ Replace with your real 32-byte AES key
  final aesKey = List<int>.generate(32, (i) => i); // placeholder!

  Future<void> runIngest() async {
    setState(() => status = 'Running pipeline...');
    final pipeline = Pipeline(aesKey32: aesKey);
    final res = await pipeline.run();
    setState(() => status =
        '${res.message} (chunks=${res.totalChunks}, refreshed=${res.refreshed})');
  }

  Future<void> runSearch() async {
    final q = queryCtrl.text.trim();
    if (q.isEmpty) return;
    setState(() => status = 'Embedding query & searching...');

    final emb = await Embeddings.load();
    final vec = emb.embed(q);
    final store = await VectorStore.open(embedSize: 384);
    final top = await store.search(vec, topK: 5);
    emb.close();

    setState(() {
      results = top.map((e) => MapEntry(e.text, e.score)).toList();
      status = 'Done';
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('On-device RAG (Embeddings + SQLite)')),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          children: [
            Text(status),
            const SizedBox(height: 12),
            Row(
              children: [
                ElevatedButton(onPressed: runIngest, child: const Text('Run Ingest')),
                const SizedBox(width: 12),
                Expanded(
                  child: TextField(
                    controller: queryCtrl,
                    decoration: const InputDecoration(
                      hintText: 'Type a query...',
                      border: OutlineInputBorder(),
                    ),
                  ),
                ),
                const SizedBox(width: 8),
                ElevatedButton(onPressed: runSearch, child: const Text('Search')),
              ],
            ),
            const SizedBox(height: 16),
            Expanded(
              child: ListView.separated(
                itemCount: results.length,
                separatorBuilder: (_, __) => const Divider(),
                itemBuilder: (_, i) {
                  final e = results[i];
                  return ListTile(
                    title: Text(e.key, maxLines: 4, overflow: TextOverflow.ellipsis),
                    subtitle: Text('cosine=${e.value.toStringAsFixed(4)}'),
                  );
                },
              ),
            )
          ],
        ),
      ),
    );
  }
}
