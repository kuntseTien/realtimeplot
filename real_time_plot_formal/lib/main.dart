import 'package:flutter/material.dart';
import 'realtime_plot.dart';
// import 'offline_test_page.dart'; // 要測 offline 再切換

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const MainApp());
}

class MainApp extends StatelessWidget {
  const MainApp({super.key});

  @override
  Widget build(BuildContext context) {
    return const MaterialApp(
      debugShowCheckedModeBanner: false,
      home: RealTimePlotPage(), // realtime 測試用
      // home: OfflineTestPage(), // offline 測試用
    );
  }
}
