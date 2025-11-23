import 'dart:math';

class DebugUtils {
  /// Analyze an embedding vector for debugging
  static Map<String, dynamic> analyzeEmbedding(List<double> embedding, {String? label}) {
    if (embedding.isEmpty) {
      return {'error': 'Empty embedding vector'};
    }
    
    final stats = <String, dynamic>{};
    
    // Basic statistics
    final sum = embedding.fold(0.0, (a, b) => a + b);
    final mean = sum / embedding.length;
    
    double variance = 0.0;
    double minVal = embedding[0];
    double maxVal = embedding[0];
    int zeroCount = 0;
    int positiveCount = 0;
    int negativeCount = 0;
    
    for (final val in embedding) {
      variance += (val - mean) * (val - mean);
      minVal = min(minVal, val);
      maxVal = max(maxVal, val);
      
      if (val == 0.0) zeroCount++;
      else if (val > 0) positiveCount++;
      else negativeCount++;
    }
    
    variance /= embedding.length;
    final stdDev = sqrt(variance);
    
    // Calculate norm
    final norm = sqrt(embedding.fold(0.0, (sum, val) => sum + val * val));
    
    stats['label'] = label ?? 'Unknown';
    stats['dimension'] = embedding.length;
    stats['mean'] = mean;
    stats['std_dev'] = stdDev;
    stats['min'] = minVal;
    stats['max'] = maxVal;
    stats['norm'] = norm;
    stats['zero_count'] = zeroCount;
    stats['positive_count'] = positiveCount;
    stats['negative_count'] = negativeCount;
    stats['zero_percentage'] = (zeroCount / embedding.length * 100);
    
    // Quality assessment
    stats['is_valid'] = norm > 1e-6;
    stats['is_normalized'] = (norm - 1.0).abs() < 0.1;
    stats['has_variation'] = stdDev > 1e-6;
    
    return stats;
  }
  
  /// Print embedding analysis in a readable format
  static void printEmbeddingAnalysis(List<double> embedding, {String? label}) {
    final stats = analyzeEmbedding(embedding, label: label);
    
    print("\n🔍 Embedding Analysis: ${stats['label']}");
    print("=" * 50);
    print("Dimension: ${stats['dimension']}");
    print("Norm: ${stats['norm']?.toStringAsFixed(6)}");
    print("Mean: ${stats['mean']?.toStringAsFixed(6)}");
    print("Std Dev: ${stats['std_dev']?.toStringAsFixed(6)}");
    print("Range: [${stats['min']?.toStringAsFixed(6)}, ${stats['max']?.toStringAsFixed(6)}]");
    print("Zero values: ${stats['zero_count']} (${stats['zero_percentage']?.toStringAsFixed(1)}%)");
    print("Positive: ${stats['positive_count']}, Negative: ${stats['negative_count']}");
    print("Valid: ${stats['is_valid']}");
    print("Normalized: ${stats['is_normalized']}");
    print("Has variation: ${stats['has_variation']}");
    
    if (embedding.length >= 10) {
      print("First 10 values: ${embedding.take(10).map((v) => v.toStringAsFixed(4)).join(', ')}");
    }
    print("");
  }
  
  /// Compare two embeddings
  static double cosineSimilarity(List<double> a, List<double> b) {
    if (a.length != b.length) {
      throw ArgumentError('Vector dimensions must match');
    }
    
    double dotProduct = 0.0;
    double normA = 0.0;
    double normB = 0.0;
    
    for (int i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }
    
    if (normA == 0.0 || normB == 0.0) {
      return 0.0;
    }
    
    return dotProduct / (sqrt(normA) * sqrt(normB));
  }
  
  /// Test embedding generation with sample texts
  static Future<void> testEmbeddingGeneration(dynamic embeddings) async {
    final testTexts = [
      "artificial intelligence",
      "machine learning algorithms", 
      "natural language processing",
      "deep learning neural networks",
      "computer vision technology"
    ];
    
    print("\n🧪 Testing Embedding Generation");
    print("=" * 50);
    
    for (int i = 0; i < testTexts.length; i++) {
      final text = testTexts[i];
      try {
        final embedding = embeddings.embedTexts([text])[0];
        printEmbeddingAnalysis(embedding, label: "Text $i: '$text'");
        
        // Test similarity with previous embeddings
        if (i > 0) {
          final prevEmbedding = embeddings.embedTexts([testTexts[i-1]])[0];
          final similarity = cosineSimilarity(embedding, prevEmbedding);
          print("Similarity with previous: ${similarity.toStringAsFixed(4)}");
        }
      } catch (e) {
        print("❌ Failed to generate embedding for '$text': $e");
      }
    }
  }
}