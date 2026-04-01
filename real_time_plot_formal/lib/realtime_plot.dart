import 'dart:async';
import 'dart:typed_data';
import 'dart:math';
import 'dart:collection';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:syncfusion_flutter_charts/charts.dart';
import 'package:path_provider/path_provider.dart';

import 'data_struct/data_point.dart';
import 'data_struct/data_processor.dart';
import 'connection/tcpip_server.dart';
import 'onnx/onnx_stand_channel.dart';

// ======================================================
// Butterworth Filter (10 Hz LP, streaming IIR)
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
// Float32 Stream Parser
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
// Model config
// ======================================================
class _ModelNormConfig {
  final double v10Sig;
  const _ModelNormConfig({required this.v10Sig});
}

const Map<String, _ModelNormConfig> _modelNormMap = {
  'STAND': _ModelNormConfig(v10Sig: 0.203235355),
  'DB': _ModelNormConfig(v10Sig: 0.118292885),
  'SLEEP': _ModelNormConfig(v10Sig: 0.082248704),
};

// ======================================================
// Realtime Plot Page
// overlap-centered output with ~2 sec delay
// baseline: 4~6 sec
// input gain: 15x
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

  // ---------- Display data ----------
  final ListQueue<DataPoint> _piezoQ = ListQueue();
  final ListQueue<DataPoint> _slmQ = ListQueue();

  // ---------- Inference buffers ----------
  final List<double> _onnxInputBuffer = [];
  final List<double> _rawVBuffer = [];
  final ListQueue<double> _recentV10Q = ListQueue();
  final ListQueue<double> _recentRawQ = ListQueue();

  final DataProcessor _dataProcessor = DataProcessor();
  TCPServer? _tcpServer;
  final _Float32StreamParser _f32Parser = _Float32StreamParser();

  // ---------- Filter / timers ----------
  final ButterworthFilter filter = ButterworthFilter();
  Timer? _xAxisTimer;
  Timer? _railTimer;
  Timer? _rateTimer;
  Timer? _flushTimer;

  bool _isInferencing = false;
  bool _onnxInitialized = false;
  bool _onnxInitializing = false;
  bool _onnxInitFailed = false;

  double xAxisMin = 0;
  double xAxisMax = 10;

  // ---------- Parameters ----------
  static const int bufferSize = 2000; // 2 sec @ 1000Hz
  static const int stride = 200; // 0.2 sec
  static const double dt = 0.001;
  static const double dropRatio = 0.05;
  static const String _eol = '\r\n';

  static const int outputStartSample = 4000; // before 4 sec no display
  static const int baselineStartSample = 4000; // 4 sec
  static const int baselineEndSample = 6000; // 6 sec

  // each window [s, s+1999] finalizes center segment [s+900, s+1099]
  static const int centerOffset = (bufferSize - stride) ~/ 2; // 900

  // >>> 新增：輸入增益 15 倍 <<<
  static const double inputGain = 1;

  int _sampleIndex = 0;
  double get _tRecord => _sampleIndex * dt;

  // ---------- Fixed baseline from 4~6 sec ----------
  bool _baselineReady = false;
  double _fixedInputBaseline = 0.0;
  double _baselineSum = 0.0;
  int _baselineCount = 0;

  // ---------- Global overlap accumulators ----------
  final List<double> _predSum = [];
  final List<double> _predWSum = [];
  int _windowStartIndex = 0;
  int _nextFinalizeIndex = 0;
  late final List<double> _windowWeights;

  // ---------- Output baseline correction ----------
  double _outputBaseline = 0.0;
  static const double outputBaselineAlpha = 0.002;
  static const double slmDeadband = 0.20;

  // ---------- Recording ----------
  bool _isRecording = false;
  Directory? _recordDir;
  File? _rawRecordFile;
  IOSink? _rawRecordSink;
  File? _predRecordFile;
  IOSink? _predRecordSink;
  final List<String> _rawPendingLines = [];
  final List<String> _predPendingLines = [];
  int _rawWriteTotal = 0;
  int _predWriteTotal = 0;
  int _rawFlushCount = 0;
  int _predFlushCount = 0;

  // ---------- Debug ----------
  int _railNear0 = 0;
  int _railNear25 = 0;
  int _railTotal = 0;
  int _rxFloatCount = 0;
  int _inferCountPerSec = 0;
  int _rawWriteCountPerSec = 0;
  int _predWriteCountPerSec = 0;
  int _inferMsAccumPerSec = 0;
  int _inferMsMaxPerSec = 0;
  int _lastInferMs = 0;

  double? _lastRawV;
  int _spikeRejectCount = 0;

  static const double displayOffset = 1.25;

  @override
  void initState() {
    super.initState();

    _windowWeights = _buildWindowWeights();

    _tcpServer = TCPServer(onDataReceived: _onTcpData);
    _tcpServer!.dataProcessor = _dataProcessor;
    _tcpServer!.start();

    _xAxisTimer = Timer.periodic(const Duration(milliseconds: 33), (_) {
      if (!mounted) return;
      setState(() {
        xAxisMin = max(0, _tRecord - 10);
        xAxisMax = _tRecord;
      });
    });

    _railTimer = Timer.periodic(const Duration(seconds: 1), (_) {
      if (_railTotal > 0) {
        final p0 = _railNear0 / _railTotal * 100.0;
        final p25 = _railNear25 / _railTotal * 100.0;

        debugPrint(
          'D/RAW_RAIL total=$_railTotal near0=${p0.toStringAsFixed(2)}% near2.5=${p25.toStringAsFixed(2)}%',
        );
        debugPrint(
          'D/BASELINE ready=$_baselineReady fixed=${_fixedInputBaseline.toStringAsFixed(6)} '
          'count=$_baselineCount sampleIndex=$_sampleIndex model=$_currentModel gain=$inputGain',
        );
        debugPrint('D/SPIKE rejected=$_spikeRejectCount / sec');
      }

      _railNear0 = 0;
      _railNear25 = 0;
      _railTotal = 0;
      _spikeRejectCount = 0;
    });

    _rateTimer = Timer.periodic(const Duration(seconds: 1), (_) {
      final avgInferMs =
          _inferCountPerSec > 0 ? _inferMsAccumPerSec / _inferCountPerSec : 0.0;

      debugPrint(
        'D/RATE rxFloat=$_rxFloatCount sampleIndex=$_sampleIndex '
        'infer=$_inferCountPerSec rawWrite=$_rawWriteCountPerSec predWrite=$_predWriteCountPerSec '
        'rawPending=${_rawPendingLines.length} predPending=${_predPendingLines.length} '
        'rawFlush=$_rawFlushCount predFlush=$_predFlushCount '
        'baselineReady=$_baselineReady nextFinalize=$_nextFinalizeIndex windowStart=$_windowStartIndex '
        'tRecord=${_tRecord.toStringAsFixed(3)} '
        'inferAvgMs=${avgInferMs.toStringAsFixed(1)} inferMaxMs=$_inferMsMaxPerSec lastInferMs=$_lastInferMs',
      );

      _rxFloatCount = 0;
      _inferCountPerSec = 0;
      _rawWriteCountPerSec = 0;
      _predWriteCountPerSec = 0;
      _inferMsAccumPerSec = 0;
      _inferMsMaxPerSec = 0;
    });

    _flushTimer = Timer.periodic(const Duration(milliseconds: 500), (_) async {
      await _flushPendingWrites();
    });
  }

  List<double> _buildWindowWeights() {
    final w = List<double>.filled(bufferSize, 1.0);
    final dropEdge = (dropRatio * bufferSize).round();
    if (dropEdge <= 0) return w;

    for (int i = 0; i < dropEdge; i++) {
      final val = dropEdge == 1 ? 1.0 : i / (dropEdge - 1);
      w[i] = val;
      w[bufferSize - dropEdge + i] = 1.0 - val;
    }
    return w;
  }

  void _ensurePredCapacity(int n) {
    while (_predSum.length < n) {
      _predSum.add(0.0);
      _predWSum.add(0.0);
    }
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
            child: _buildChart(
              'Piezo (~1.25 V display)',
              _piezoQ,
              yMin: 1.15,
              yMax: 1.35,
            ),
          ),
          Expanded(
            child: _buildChart(
              'Predicted Flow (Overlap-Centered, ~2s delay)',
              _slmQ,
              yMin: -8,
              yMax: 8,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildChart(
    String title,
    ListQueue<DataPoint> data, {
    required double yMin,
    required double yMax,
  }) {
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

  List<DataPoint> _onTcpData(List<int> bytes) {
    final f32 = _f32Parser.push(bytes);
    if (f32.isEmpty) return [];

    _rxFloatCount += f32.length;

    for (int i = 0; i < f32.length; i++) {
      _consumeOneSample(f32[i].toDouble());
    }

    _trimQueue(_piezoQ, _tRecord - 10);
    _runOnnxIfReady();
    return [];
  }

  void _consumeOneSample(double rawV) {
    final idx = _sampleIndex;

    _railTotal++;
    if (rawV.abs() <= 0.01) _railNear0++;
    if (rawV >= 2.49) _railNear25++;

    double rawUsed = rawV;
    if (_lastRawV != null) {
      final diff = (rawV - _lastRawV!).abs();
      if (diff > 0.30) {
        rawUsed = _lastRawV!;
        _spikeRejectCount++;
      }
    }
    _lastRawV = rawUsed;

    final v10 = filter.apply(rawUsed);

    _recentV10Q.addLast(v10);
    _recentRawQ.addLast(rawUsed);
    while (_recentV10Q.length > bufferSize) {
      _recentV10Q.removeFirst();
    }
    while (_recentRawQ.length > bufferSize) {
      _recentRawQ.removeFirst();
    }

    if (idx >= baselineStartSample && idx < baselineEndSample) {
      _baselineSum += v10;
      _baselineCount++;
    }

    if (!_baselineReady && idx == baselineEndSample - 1) {
      _fixedInputBaseline =
          _baselineCount > 0 ? (_baselineSum / _baselineCount) : 0.0;
      _baselineReady = true;

      _onnxInputBuffer.clear();
      _rawVBuffer.clear();

      final cfg = _modelNormMap[_currentModel]!;
      final recentV10 = _recentV10Q.toList(growable: false);
      final recentRaw = _recentRawQ.toList(growable: false);

      for (int i = 0; i < recentV10.length; i++) {
        final v10m0 = recentV10[i] - _fixedInputBaseline;
        final v10m0Gain = v10m0 * inputGain;
        final v10Norm = v10m0Gain / cfg.v10Sig;
        _onnxInputBuffer.add(v10Norm);
        _rawVBuffer.add(recentRaw[i]);
      }

      _windowStartIndex = baselineStartSample;
      _nextFinalizeIndex = baselineStartSample + centerOffset;

      debugPrint(
        'D/BASELINE fixed from 4~6s = ${_fixedInputBaseline.toStringAsFixed(6)}, '
        'backfill=${_onnxInputBuffer.length}, windowStart=$_windowStartIndex, gain=$inputGain',
      );
    }

    double displayBaseline;
    if (_baselineReady) {
      displayBaseline = _fixedInputBaseline;
    } else if (_baselineCount > 0) {
      displayBaseline = _baselineSum / _baselineCount;
    } else {
      displayBaseline = v10;
    }

    final v10DisplayM0 = v10 - displayBaseline;
    final t = idx * dt;
    final displayV = displayOffset + v10DisplayM0;

    if (idx >= outputStartSample) {
      _piezoQ.addLast(DataPoint(t, displayV, rawUsed));
    }

    if (_baselineReady && idx >= baselineEndSample) {
      final cfg = _modelNormMap[_currentModel]!;
      final v10m0 = v10 - _fixedInputBaseline;
      final v10m0Gain = v10m0 * inputGain;
      final v10Norm = v10m0Gain / cfg.v10Sig;
      _onnxInputBuffer.add(v10Norm);
      _rawVBuffer.add(rawUsed);
    }

    if (_isRecording) {
      final cfg = _modelNormMap[_currentModel]!;
      final v10MeanCsv = _baselineReady ? _fixedInputBaseline : double.nan;
      final v10m0Csv =
          _baselineReady ? (v10 - _fixedInputBaseline) : double.nan;
      final v10m0GainCsv = _baselineReady ? (v10m0Csv * inputGain) : double.nan;
      final v10NormCsv =
          _baselineReady ? (v10m0GainCsv / cfg.v10Sig) : double.nan;

      _enqueueRawWrite(
        '$t,$rawV,$rawUsed,$v10,$v10MeanCsv,$v10m0Csv,$v10NormCsv',
      );
      _rawWriteCountPerSec++;
      _rawWriteTotal++;
    }

    _sampleIndex++;
  }

  void _runOnnxIfReady() {
    if (!_baselineReady) return;
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
        debugPrint('ONNX init success: $_currentModel');
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

    final windowStart = _windowStartIndex;
    final windowInput = _onnxInputBuffer.sublist(0, bufferSize);

    final input = Float32List(bufferSize);
    for (int i = 0; i < bufferSize; i++) {
      input[i] = windowInput[i];
    }

    final sw = Stopwatch()..start();

    OnnxStandChannel.infer(input).then((out) {
      sw.stop();
      final inferMs = sw.elapsedMilliseconds;
      _lastInferMs = inferMs;
      _inferCountPerSec++;
      _inferMsAccumPerSec += inferMs;
      if (inferMs > _inferMsMaxPerSec) _inferMsMaxPerSec = inferMs;

      debugPrint(
        'D/INFER model=$_currentModel start=$windowStart inLen=${input.length} outLen=${out.length} '
        'stride=$stride inferMs=$inferMs sampleIndex=$_sampleIndex',
      );

      if (out.length < bufferSize) {
        debugPrint('D/INFER skip: invalid output length = ${out.length}');
        return;
      }

      final neededLen = windowStart + bufferSize;
      _ensurePredCapacity(neededLen);

      for (int i = 0; i < bufferSize; i++) {
        final gi = windowStart + i;
        final w = _windowWeights[i];
        final y = out[i].toDouble();
        _predSum[gi] += y * w;
        _predWSum[gi] += w;
      }

      final finalizeStart = windowStart + centerOffset;
      final finalizeEndExclusive = finalizeStart + stride;

      _finalizeCenteredSegment(finalizeStart, finalizeEndExclusive, inferMs);

      _onnxInputBuffer.removeRange(0, stride);
      _rawVBuffer.removeRange(0, stride);
      _windowStartIndex += stride;

      _trimQueue(_slmQ, _tRecord - 10);
      if (mounted) setState(() {});
    }).catchError((e) {
      sw.stop();
      debugPrint('ONNX infer error: $e');
    }).whenComplete(() {
      _isInferencing = false;
      if (_onnxInputBuffer.length >= bufferSize) {
        _runOnnxIfReady();
      }
    });
  }

  void _finalizeCenteredSegment(
      int startInclusive, int endExclusive, int inferMs) {
    while (_nextFinalizeIndex < endExclusive &&
        _nextFinalizeIndex >= startInclusive &&
        _nextFinalizeIndex < _predSum.length &&
        _nextFinalizeIndex < _predWSum.length) {
      final w = _predWSum[_nextFinalizeIndex];
      if (w <= 0) {
        _nextFinalizeIndex++;
        continue;
      }

      final slmOla = _predSum[_nextFinalizeIndex] / w;

      _outputBaseline = (1.0 - outputBaselineAlpha) * _outputBaseline +
          outputBaselineAlpha * slmOla;

      double slmCorr = slmOla - _outputBaseline;
      if (slmCorr.abs() < slmDeadband) {
        slmCorr = 0.0;
      }

      final t = _nextFinalizeIndex * dt;

      if (_nextFinalizeIndex >= outputStartSample) {
        _slmQ.addLast(DataPoint(t, slmCorr, slmCorr));

        if (_isRecording) {
          _enqueuePredWrite('$t,$slmOla,$slmCorr,$w,$inferMs');
          _predWriteCountPerSec++;
          _predWriteTotal++;
        }
      }

      _nextFinalizeIndex++;
    }
  }

  void _finalizeTailForStop() {
    final endExclusive = min(_predSum.length, _sampleIndex);
    while (_nextFinalizeIndex < endExclusive) {
      final w = _predWSum[_nextFinalizeIndex];
      if (w <= 0) {
        _nextFinalizeIndex++;
        continue;
      }

      final slmOla = _predSum[_nextFinalizeIndex] / w;
      _outputBaseline = (1.0 - outputBaselineAlpha) * _outputBaseline +
          outputBaselineAlpha * slmOla;

      double slmCorr = slmOla - _outputBaseline;
      if (slmCorr.abs() < slmDeadband) {
        slmCorr = 0.0;
      }

      final t = _nextFinalizeIndex * dt;
      if (_nextFinalizeIndex >= outputStartSample) {
        _slmQ.addLast(DataPoint(t, slmCorr, slmCorr));

        if (_isRecording) {
          _enqueuePredWrite('$t,$slmOla,$slmCorr,$w,$_lastInferMs');
          _predWriteCountPerSec++;
          _predWriteTotal++;
        }
      }
      _nextFinalizeIndex++;
    }
  }

  void _switchModel(String newModel) {
    _currentModel = newModel;

    _onnxInitialized = false;
    _onnxInitializing = false;
    _onnxInitFailed = false;

    _onnxInputBuffer.clear();
    _rawVBuffer.clear();
    _piezoQ.clear();
    _slmQ.clear();

    _recentV10Q.clear();
    _recentRawQ.clear();

    _predSum.clear();
    _predWSum.clear();
    _windowStartIndex = 0;
    _nextFinalizeIndex = 0;

    _baselineReady = false;
    _fixedInputBaseline = 0.0;
    _baselineSum = 0.0;
    _baselineCount = 0;

    _outputBaseline = 0.0;

    _f32Parser.clear();
    filter.reset();

    _lastRawV = null;
    _spikeRejectCount = 0;
    _sampleIndex = 0;

    setState(() {});
  }

  Future<void> _toggleRecording() async {
    if (_isRecording) {
      await _stopRecording();
      return;
    }
    await _startRecording();
  }

  Future<void> _startRecording() async {
    _onnxInputBuffer.clear();
    _rawVBuffer.clear();
    _piezoQ.clear();
    _slmQ.clear();

    _recentV10Q.clear();
    _recentRawQ.clear();

    _predSum.clear();
    _predWSum.clear();
    _windowStartIndex = 0;
    _nextFinalizeIndex = 0;

    _baselineReady = false;
    _fixedInputBaseline = 0.0;
    _baselineSum = 0.0;
    _baselineCount = 0;

    _outputBaseline = 0.0;

    _f32Parser.clear();
    filter.reset();

    _lastRawV = null;
    _spikeRejectCount = 0;
    _sampleIndex = 0;

    _rawPendingLines.clear();
    _predPendingLines.clear();

    _rawWriteTotal = 0;
    _predWriteTotal = 0;
    _rawFlushCount = 0;
    _predFlushCount = 0;

    _recordDir = await _prepareRecordDirectory();

    final ts = DateTime.now().toIso8601String().replaceAll(':', '-');

    _rawRecordFile = File('${_recordDir!.path}/raw_${_currentModel}_$ts.csv');
    _predRecordFile = File('${_recordDir!.path}/pred_${_currentModel}_$ts.csv');

    _rawRecordSink = _rawRecordFile!.openWrite(mode: FileMode.writeOnlyAppend);
    _predRecordSink =
        _predRecordFile!.openWrite(mode: FileMode.writeOnlyAppend);

    _rawRecordSink!.write(
      't,rawV_original,rawV_used,v10,v10Mean,v10m0,v10Norm$_eol',
    );
    _predRecordSink!.write(
      't,slmOla,slmCorr,wSum,inferMs$_eol',
    );

    setState(() => _isRecording = true);

    debugPrint('REC START model=$_currentModel');
    debugPrint('REC DIR=${_recordDir!.path}');
    debugPrint('RAW path=${_rawRecordFile!.path}');
    debugPrint('PRED path=${_predRecordFile!.path}');
  }

  Future<void> _stopRecording() async {
    _finalizeTailForStop();
    await _flushPendingWrites(force: true);

    try {
      await _rawRecordSink?.flush();
      await _rawRecordSink?.close();
    } catch (e) {
      debugPrint('RAW close error: $e');
    }

    try {
      await _predRecordSink?.flush();
      await _predRecordSink?.close();
    } catch (e) {
      debugPrint('PRED close error: $e');
    }

    final rawSavedPath = _rawRecordFile?.path;
    final predSavedPath = _predRecordFile?.path;

    _rawRecordSink = null;
    _predRecordSink = null;
    _rawRecordFile = null;
    _predRecordFile = null;

    if (mounted) {
      setState(() => _isRecording = false);
    }

    debugPrint(
      'REC STOP sampleIndex=$_sampleIndex '
      'tRecord=${_tRecord.toStringAsFixed(3)} '
      'baselineReady=$_baselineReady fixedBaseline=${_fixedInputBaseline.toStringAsFixed(6)} '
      'rawWriteTotal=$_rawWriteTotal predWriteTotal=$_predWriteTotal '
      'rawFlush=$_rawFlushCount predFlush=$_predFlushCount '
      'nextFinalize=$_nextFinalizeIndex '
      'rawPending=${_rawPendingLines.length} predPending=${_predPendingLines.length}',
    );

    if (rawSavedPath != null) {
      debugPrint('RAW Saved: $rawSavedPath');
    }
    if (predSavedPath != null) {
      debugPrint('PRED Saved: $predSavedPath');
    }
  }

  Future<Directory> _prepareRecordDirectory() async {
    final baseDir = await getApplicationDocumentsDirectory();
    final dir = Directory('${baseDir.path}/realtime_records');
    if (!await dir.exists()) {
      await dir.create(recursive: true);
    }
    return dir;
  }

  void _enqueueRawWrite(String line) {
    _rawPendingLines.add(line);
  }

  void _enqueuePredWrite(String line) {
    _predPendingLines.add(line);
  }

  Future<void> _flushPendingWrites({bool force = false}) async {
    if (!_isRecording && !force) return;

    try {
      if (_rawPendingLines.isNotEmpty && _rawRecordSink != null) {
        _rawRecordSink!.write(_rawPendingLines.join(_eol));
        _rawRecordSink!.write(_eol);
        _rawPendingLines.clear();
        _rawFlushCount++;
      }

      if (_predPendingLines.isNotEmpty && _predRecordSink != null) {
        _predRecordSink!.write(_predPendingLines.join(_eol));
        _predRecordSink!.write(_eol);
        _predPendingLines.clear();
        _predFlushCount++;
      }

      await _rawRecordSink?.flush();
      await _predRecordSink?.flush();
    } catch (e) {
      debugPrint('FLUSH write error: $e');
      if (mounted) {
        setState(() => _isRecording = false);
      }
    }
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
    _rateTimer?.cancel();
    _flushTimer?.cancel();
    _tcpServer?.stop();
    _dataProcessor.dispose();

    () async {
      _finalizeTailForStop();
      await _flushPendingWrites(force: true);
      try {
        await _rawRecordSink?.flush();
        await _rawRecordSink?.close();
      } catch (_) {}
      try {
        await _predRecordSink?.flush();
        await _predRecordSink?.close();
      } catch (_) {}
    }();

    super.dispose();
  }
}
