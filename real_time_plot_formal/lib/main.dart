import 'package:flutter/material.dart';
import 'realtime_plot.dart';
import 'package:permission_handler/permission_handler.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized(); // 確保Flutter绑定初始化
  await requestPermissions(); // 請求權限
  runApp(const MainApp());
}

Future<void> requestPermissions() async {
  await Permission.storage.request();
}

class MainApp extends StatelessWidget {
  const MainApp({super.key});

  @override
  Widget build(BuildContext context) {
    return const MaterialApp(
      home: RealTimePlotPage(initialDataLength: 3000),
    );
  }
}
