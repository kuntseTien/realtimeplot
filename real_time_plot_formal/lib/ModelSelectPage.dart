import 'package:flutter/material.dart';
import 'realtime_plot.dart';

class ModelSelectPage extends StatefulWidget {
  const ModelSelectPage({super.key});

  @override
  State<ModelSelectPage> createState() => _ModelSelectPageState();
}

class _ModelSelectPageState extends State<ModelSelectPage> {
  String selected = "STAND";

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('選擇模型')),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          children: [
            DropdownButtonFormField<String>(
              value: selected,
              items: const [
                DropdownMenuItem(value: "STAND", child: Text("STAND")),
                DropdownMenuItem(value: "DB", child: Text("DB")),
                DropdownMenuItem(value: "SLEEP", child: Text("SLEEP")),
              ],
              onChanged: (v) => setState(() => selected = v ?? "STAND"),
              decoration: const InputDecoration(
                labelText: "Model",
                border: OutlineInputBorder(),
              ),
            ),
            const SizedBox(height: 16),
            SizedBox(
              width: double.infinity,
              child: ElevatedButton(
                onPressed: () {},
                child: const Text("開始測試"),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
