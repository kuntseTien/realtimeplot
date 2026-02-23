import 'dart:async';
import 'dart:typed_data';
import 'dart:math';
import 'dart:collection';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:syncfusion_flutter_charts/charts.dart';

import 'data_struct/data_point.dart';
import 'data_struct/data_processor.dart';
import 'connection/tcpip_server.dart';
import 'onnx/onnx_stand_channel.dart';

// ======================================================
// Butterworth Filter (10 Hz LP, same as MATLAB)
// ======================================================
class ButterworthFilter {
  final List<List<double>> sosCoefficients = [
    [1, 2, 1, -1.975269634851873, 0.97624479235944],
    [1, 2, 1, -1.9426382305401135, 0.9435972784703671],
  ];

  final List<double> gains = [
    0.00024378937689168925,
    0.00023976198256338974,
  ];

  final List<List<double>> states = List.generate(2, (_) => [0.0, 0.0]);

  double apply(double input) {
    double x = input;
    for (int i = 0; i < sosCoefficients.length; i++) {
      final b = sosCoefficients[i];
      final s = states[i];
      final v = x - b[3] * s[0] - b[4] * s[1];
      final y = gains[i] * (b[0] * v + b[1] * s[0] + b[2] * s[1]);
      s[1] = s[0];
      s[0] = v;
      x = y;
    }
    return x;
  }

  void reset() {
    for (final s in states) {
      s[0] = 0.0;
      s[1] = 0.0;
    }
  }
}

// ======================================================
// ✅ Float32 Stream Parser (fix TCP chunk boundary issue)
// ======================================================
class _Float32StreamParser {
  Uint8List _buf = Uint8List(0);

  Float32List push(List<int> chunk) {
    if (chunk.isEmpty) return Float32List(0);

    final incoming = Uint8List.fromList(chunk);
    if (_buf.isEmpty) {
      _buf = incoming;
    } else {
      final merged = Uint8List(_buf.length + incoming.length);
      merged.setAll(0, _buf);
      merged.setAll(_buf.length, incoming);
      _buf = merged;
    }

    final nBytes = _buf.length;
    final nFloats = nBytes ~/ 4;
    if (nFloats <= 0) return Float32List(0);

    final usableBytes = nFloats * 4;
    final usable = Uint8List.sublistView(_buf, 0, usableBytes);
    final floats = usable.buffer.asFloat32List();

    final rem = nBytes - usableBytes;
    if (rem > 0) {
      _buf = Uint8List.sublistView(_buf, usableBytes, nBytes);
    } else {
      _buf = Uint8List(0);
    }
    return floats;
  }

  void clear() {
    _buf = Uint8List(0);
  }
}

// ======================================================
// RealTimePlotPage
// ======================================================
class RealTimePlotPage extends StatefulWidget {
  const RealTimePlotPage({Key? key}) : super(key: key);

  @override
  State<RealTimePlotPage> createState() => _RealTimePlotState();
}

class _RealTimePlotState extends State<RealTimePlotPage> {
  // ---------- Model ----------
  String _currentModel = 'STAND';
  String? _pendingModel;

  // ---------- Data ----------
  final ListQueue<DataPoint> _piezoQ = ListQueue();
  final ListQueue<DataPoint> _slmQ = ListQueue();

  // ✅ 推論 input：v10
  final List<double> _onnxInputBuffer = [];

  // ✅ 新增：逐筆對齊 rawV（用於錄檔）
  final List<double> _rawVBuffer = [];

  final DataProcessor _dataProcessor = DataProcessor();
  TCPServer? _tcpServer;

  final _Float32StreamParser _f32Parser = _Float32StreamParser();
  double? _pendingSecondChannelPairFirst;

  // ---------- State ----------
  final ButterworthFilter filter = ButterworthFilter();
  Timer? _xAxisTimer;

  bool _isInferencing = false;
  bool _onnxInitialized = false;
  bool _onnxInitializing = false;
  bool _onnxInitFailed = false;

  double xAxisMin = 0;
  double xAxisMax = 10;

  static const int bufferSize = 2000;
  static const int stride = 200;
  static const double dt = 0.001;

  int _sampleIndex = 0;
  double get _tRecord => _sampleIndex * dt;

  // ---------- Recording ----------
  bool _isRecording = false;
  File? _recordFile;
  IOSink? _recordSink;
  Future<void> _writeQueue = Future.value();

  // rail debug
  int _railNear0 = 0;
  int _railNear25 = 0;
  int _railTotal = 0;
  Timer? _railTimer;

  @override
  void initState() {
    super.initState();

    _tcpServer = TCPServer(onDataReceived: _onTcpData);
    _tcpServer!.dataProcessor = _dataProcessor;
    _tcpServer!.start();

    _xAxisTimer = Timer.periodic(const Duration(milliseconds: 33), (_) {
      setState(() {
        xAxisMin = max(0, _tRecord - 10);
        xAxisMax = _tRecord;
      });
    });

    _railTimer = Timer.periodic(const Duration(seconds: 1), (_) {
      if (_railTotal <= 0) return;
      final p0 = _railNear0 / _railTotal * 100.0;
      final p25 = _railNear25 / _railTotal * 100.0;
      debugPrint(
          'D/RAW_RAIL total=$_railTotal near0=${p0.toStringAsFixed(1)}% near2.5=${p25.toStringAsFixed(1)}%');
      _railNear0 = 0;
      _railNear25 = 0;
      _railTotal = 0;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Row(
          children: [
            const Text('Realtime Plot'),
            const SizedBox(width: 12),
            DropdownButton<String>(
              value: _currentModel,
              underline: const SizedBox(),
              items: const [
                DropdownMenuItem(value: 'STAND', child: Text('STAND')),
                DropdownMenuItem(value: 'DB', child: Text('DB')),
                DropdownMenuItem(value: 'SLEEP', child: Text('SLEEP')),
              ],
              onChanged: (v) {
                if (v != null && v != _currentModel) {
                  setState(() => _pendingModel = v);
                }
              },
            ),
            const Spacer(),
            ElevatedButton(
              onPressed: _toggleRecording,
              child: Text(_isRecording ? 'Stop REC' : 'Start REC'),
            ),
          ],
        ),
      ),
      body: Column(
        children: [
          Expanded(
            child: _buildChart('Piezo (10 Hz LP)', _piezoQ, yMin: -2, yMax: 4),
          ),
          Expanded(
            child:
                _buildChart('Predicted Flow (SLM)', _slmQ, yMin: -8, yMax: 8),
          ),
        ],
      ),
    );
  }

  Widget _buildChart(String title, ListQueue<DataPoint> data,
      {required double yMin, required double yMax}) {
    return SfCartesianChart(
      title: ChartTitle(text: title),
      primaryXAxis: NumericAxis(minimum: xAxisMin, maximum: xAxisMax),
      primaryYAxis: NumericAxis(minimum: yMin, maximum: yMax),
      series: [
        FastLineSeries<DataPoint, double>(
          dataSource: data.toList(),
          xValueMapper: (d, _) => d.t,
          yValueMapper: (d, _) => d.y,
        ),
      ],
    );
  }

  // ======================================================
  // TCP handler
  // ======================================================
  List<DataPoint> _onTcpData(List<int> bytes) {
    final f32 = _f32Parser.push(bytes);
    if (f32.isEmpty) return [];

    int i = 0;

    if (_pendingSecondChannelPairFirst != null) {
      final rawV = _pendingSecondChannelPairFirst!;
      final other = f32[0].toDouble();
      _pendingSecondChannelPairFirst = null;
      _consumeOneSample(rawV, other);
      i = 1;
    }

    for (; i + 1 < f32.length; i += 2) {
      final rawV = f32[i].toDouble();
      final other = f32[i + 1].toDouble();
      _consumeOneSample(rawV, other);
    }

    if (i < f32.length) {
      _pendingSecondChannelPairFirst = f32[i].toDouble();
    }

    _trimQueue(_piezoQ, _tRecord - 10);
    _runOnnxIfReady();
    return [];
  }

  void _consumeOneSample(double rawV, double otherChannel) {
    _railTotal++;
    if (rawV.abs() <= 0.01) _railNear0++;
    if (rawV >= 2.49) _railNear25++;

    final v10 = filter.apply(rawV);
    final t = _sampleIndex * dt;

    _piezoQ.addLast(DataPoint(t, v10, rawV));

    // ✅ input buffer + raw buffer 都存，確保逐筆對齊
    _onnxInputBuffer.add(v10);
    _rawVBuffer.add(rawV);

    _sampleIndex++;
  }

  // ======================================================
  // ONNX logic
  // ======================================================
  void _runOnnxIfReady() {
    if (_onnxInitFailed || _isInferencing || _onnxInitializing) return;
    if (_onnxInputBuffer.length < bufferSize) return;

    if (_pendingModel != null) {
      _switchModel(_pendingModel!);
      _pendingModel = null;
      return;
    }

    if (!_onnxInitialized) {
      _onnxInitializing = true;
      _isInferencing = true;

      OnnxStandChannel.init(_currentModel).then((_) {
        _onnxInitialized = true;
      }).catchError((e) {
        _onnxInitFailed = true;
        debugPrint('ONNX init error: $e');
      }).whenComplete(() {
        _onnxInitializing = false;
        _isInferencing = false;
      });
      return;
    }

    _isInferencing = true;

    // window snapshot（避免 async 回來時 buffer 已經被推進）
    final windowV10 = _onnxInputBuffer.sublist(0, bufferSize);
    final windowRaw = _rawVBuffer.sublist(0, bufferSize);

    final input = Float32List(bufferSize);
    for (int i = 0; i < bufferSize; i++) {
      input[i] = windowV10[i];
    }

    final lastStrideV10 = windowV10.sublist(bufferSize - stride, bufferSize);
    final lastStrideRaw = windowRaw.sublist(bufferSize - stride, bufferSize);
    final firstIdx = _sampleIndex - stride;

    // 推進 buffer
    _onnxInputBuffer.removeRange(0, stride);
    _rawVBuffer.removeRange(0, stride);

    OnnxStandChannel.infer(input).then((out) {
      final startIdx = out.length - stride;

      for (int i = 0; i < stride; i++) {
        final idx = firstIdx + i;
        final t = idx * dt;
        final slm = out[startIdx + i];

        _slmQ.addLast(DataPoint(t, slm, slm));

        if (_isRecording) {
          final rawV = lastStrideRaw[i];
          final v10 = lastStrideV10[i];
          _enqueueWrite('$t,$rawV,$v10,$slm');
        }
      }

      _trimQueue(_slmQ, _tRecord - 10);
      setState(() {});
    }).catchError((e) {
      debugPrint('ONNX infer error: $e');
    }).whenComplete(() {
      _isInferencing = false;
    });
  }

  void _switchModel(String newModel) {
    _currentModel = newModel;

    _onnxInitialized = false;
    _onnxInitializing = false;
    _onnxInitFailed = false;

    _onnxInputBuffer.clear();
    _rawVBuffer.clear();
    _slmQ.clear();
    _piezoQ.clear();

    _pendingSecondChannelPairFirst = null;
    _f32Parser.clear();
    filter.reset();

    setState(() {});
  }

  // ======================================================
  // Recording
  // ======================================================
  Future<void> _toggleRecording() async {
    if (_isRecording) {
      try {
        await _writeQueue;
        await _recordSink?.flush();
        await _recordSink?.close();
      } catch (_) {}

      final savedPath = _recordFile?.path;
      _recordSink = null;
      _recordFile = null;

      setState(() => _isRecording = false);

      if (savedPath != null) {
        debugPrint('✅ Saved: $savedPath');
      }
      return;
    }

    final dir = Directory('/storage/emulated/0/Download');
    if (!await dir.exists()) {
      await dir.create(recursive: true);
    }

    final ts = DateTime.now().toIso8601String().replaceAll(':', '-');
    _recordFile = File('${dir.path}/record_${_currentModel}_$ts.csv');

    _recordSink = _recordFile!.openWrite();
    _recordSink!.writeln('t,rawV,v10,slm');

    setState(() => _isRecording = true);
  }

  void _enqueueWrite(String line) {
    _writeQueue = _writeQueue.then((_) async {
      try {
        _recordSink?.writeln(line);
      } catch (e) {
        debugPrint('Record write error: $e');
        try {
          await _recordSink?.flush();
          await _recordSink?.close();
        } catch (_) {}
        _recordSink = null;
        _recordFile = null;
        if (mounted) setState(() => _isRecording = false);
      }
    });
  }

  void _trimQueue(ListQueue<DataPoint> q, double minT) {
    while (q.isNotEmpty && q.first.t < minT) {
      q.removeFirst();
    }
  }

  @override
  void dispose() {
    _xAxisTimer?.cancel();
    _railTimer?.cancel();
    _tcpServer?.stop();
    _dataProcessor.dispose();

    () async {
      try {
        await _writeQueue;
        await _recordSink?.flush();
        await _recordSink?.close();
      } catch (_) {}
    }();

    super.dispose();
  }
}
