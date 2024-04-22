import 'package:flutter/material.dart';

import 'realtime_plot.dart';

void main() {
  runApp(const MainApp());
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
