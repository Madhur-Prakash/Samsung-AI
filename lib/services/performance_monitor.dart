import 'dart:io';

class PerformanceMonitor {
  static final Map<String, DateTime> _startTimes = {};
  static final Map<String, List<int>> _durations = {};
  
  static void startTimer(String operation) {
    _startTimes[operation] = DateTime.now();
    print("⏱️ Started: $operation");
  }
  
  static void endTimer(String operation) {
    final startTime = _startTimes[operation];
    if (startTime == null) {
      print("⚠️ No start time found for operation: $operation");
      return;
    }
    
    final duration = DateTime.now().difference(startTime).inMilliseconds;
    _durations.putIfAbsent(operation, () => []).add(duration);
    
    print("✅ Completed: $operation in ${duration}ms");
    _startTimes.remove(operation);
  }
  
  static void logMemoryUsage(String context) {
    try {
      final info = ProcessInfo.currentRss;
      final memoryMB = (info / (1024 * 1024)).toStringAsFixed(1);
      print("🧠 Memory usage at $context: ${memoryMB}MB");
    } catch (e) {
      print("⚠️ Could not get memory info: $e");
    }
  }
  
  static void printSummary() {
    print("\n📊 Performance Summary:");
    print("=" * 50);
    
    for (final entry in _durations.entries) {
      final operation = entry.key;
      final times = entry.value;
      
      if (times.isNotEmpty) {
        final avg = times.reduce((a, b) => a + b) / times.length;
        final min = times.reduce((a, b) => a < b ? a : b);
        final max = times.reduce((a, b) => a > b ? a : b);
        
        print("$operation:");
        print("  Calls: ${times.length}");
        print("  Average: ${avg.toStringAsFixed(1)}ms");
        print("  Min: ${min}ms, Max: ${max}ms");
        print("");
      }
    }
  }
  
  static void reset() {
    _startTimes.clear();
    _durations.clear();
    print("🔄 Performance monitor reset");
  }
}