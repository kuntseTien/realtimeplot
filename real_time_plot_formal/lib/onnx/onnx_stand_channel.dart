import 'dart:typed_data';
import 'package:flutter/services.dart';

class OnnxStandChannel {
  static const MethodChannel _channel = MethodChannel('onnx_stand_channel');

  static bool _initialized = false;
  static String? _currentModel;

  static Future<void> init(String modelName) async {
    if (_initialized && _currentModel == modelName) return;

    await _channel.invokeMethod('init', {
      'model': modelName,
    });

    _initialized = true;
    _currentModel = modelName;
  }

  static Future<Float32List> infer(Float32List input) async {
    if (!_initialized) {
      throw StateError(
          'OnnxStandChannel not initialized. Call init(model) first.');
    }

    final List<dynamic> raw = await _channel.invokeMethod<List<dynamic>>(
          'infer',
          {'input': input.map((e) => e.toDouble()).toList()},
        ) ??
        [];

    final output = Float32List(raw.length);
    for (int i = 0; i < raw.length; i++) {
      output[i] = (raw[i] as num).toDouble();
    }
    return output;
  }
}
