// lib/offline_test_page.dart
import 'dart:async';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:syncfusion_flutter_charts/charts.dart';

import 'onnx/onnx_stand_channel.dart';
import 'overlap_add_inferencer.dart';

class DataPoint {
  final double t;
  final double y;
  DataPoint(this.t, this.y);
}

/// ======================================================
/// MATLAB-aligned Butterworth(2) 10Hz + filtfilt
/// MATLAB:
///   [b10,a10] = butter(2,10/(1000/2));
///   b10 = [0.0009 0.0019 0.0009]
///   a10 = [1.0000 -1.9112 0.9150]
/// ======================================================
class Butterworth2Filtfilt10Hz {
  static const List<double> b = [0.0009, 0.0019, 0.0009];
  static const List<double> a = [1.0, -1.9112, 0.9150];

  static Float64List _lfilter(Float64List x) {
    final y = Float64List(x.length);
    double z1 = 0.0, z2 = 0.0;

    final b0 = b[0], b1 = b[1], b2 = b[2];
    final a1 = a[1], a2 = a[2];

    for (int n = 0; n < x.length; n++) {
      final xn = x[n];
      final yn = b0 * xn + z1;
      z1 = b1 * xn - a1 * yn + z2;
      z2 = b2 * xn - a2 * yn;
      y[n] = yn;
    }
    return y;
  }

  static Float64List _reverse(Float64List x) {
    final y = Float64List(x.length);
    for (int i = 0; i < x.length; i++) {
      y[i] = x[x.length - 1 - i];
    }
    return y;
  }

  static Float64List _mirrorPad(Float64List x, int pad) {
    final n = x.length;
    final p = math.min(pad, n - 1);
    final out = Float64List(n + 2 * p);

    // left
    for (int i = 0; i < p; i++) out[i] = x[p - i];
    // center
    for (int i = 0; i < n; i++) out[p + i] = x[i];
    // right
    for (int i = 0; i < p; i++) out[p + n + i] = x[n - 2 - i];

    return out;
  }

  static Float64List _unpad(Float64List x, int pad, int originalLen) {
    final p = math.min(pad, originalLen - 1);
    final y = Float64List(originalLen);
    for (int i = 0; i < originalLen; i++) y[i] = x[p + i];
    return y;
  }

  static Float64List filtfilt(Float64List x, {int pad = 300}) {
    if (x.length < 3) return Float64List.fromList(x);
    final padded = _mirrorPad(x, pad);
    final fwd = _lfilter(padded);
    final rev1 = _reverse(fwd);
    final bwd = _lfilter(rev1);
    final rev2 = _reverse(bwd);
    return _unpad(rev2, pad, x.length);
  }
}

class OfflineTestPage extends StatefulWidget {
  const OfflineTestPage({super.key});

  @override
  State<OfflineTestPage> createState() => _OfflineTestPageState();
}

class _OfflineTestPageState extends State<OfflineTestPage> {
  // ---- Config ----
  static const int fs = 1000;
  static const int startIdx = 4000;
  static const int endIdx = 74000;

  static const double VDD = 5.0;
  static const double V_ZERO = 0.74;

  static const int winSize = 2000;
  static const int stride = 200;
  static const double dropRatio = 0.05;

  // ---- Selectors ----
  final List<String> _models = const ['STAND', 'DB', 'SLEEP'];

  final Map<String, String> _assetCsvByModel = const {
    'STAND': 'assets/testdata/chirustand03.csv',
    'DB': 'assets/testdata/leeDB02.csv',
    'SLEEP': 'assets/testdata/tiensleep01.csv',
  };

  String _currentModel = 'STAND';

  // ---- State ----
  bool _running = false;
  bool _saving = false;
  bool _hasResult = false;
  String _log = '';

  // ---- UI Data ----
  final List<DataPoint> _vRawPlot = [];
  final List<DataPoint> _v10Plot = [];
  final List<DataPoint> _predPlot = [];
  final List<DataPoint> _truePlot = [];

  // ---- Results to save ----
  Float64List? _t;
  Float64List? _vRaw;
  Float64List? _v10Full; // v10 filtfilt + mean0
  Float64List? _trueSlm; // true flow filtfilt + mean0
  Float64List? _predSlm; // pred SLM, then mean0 (MATLAB-aligned)

  late OverlapAddInferencer _inferencer;

  @override
  void initState() {
    super.initState();
    _initModel(_currentModel);
  }

  Future<void> _initModel(String modelName) async {
    await OnnxStandChannel.init(modelName);

    _inferencer = OverlapAddInferencer(
      winSize: winSize,
      stride: stride,
      dropRatio: dropRatio,
      inferFn: (Float32List w) => OnnxStandChannel.infer(w),
    );

    if (!mounted) return;
    setState(() {
      _currentModel = modelName;
    });
  }

  String get _currentAssetCsv => _assetCsvByModel[_currentModel] ?? '';

  String _basenameNoExt(String assetPath) {
    final name = assetPath.split('/').last;
    final dot = name.lastIndexOf('.');
    return (dot > 0) ? name.substring(0, dot) : name;
  }

  void _appendLog(String s) {
    // ignore: avoid_print
    print(s);
    if (!mounted) return;
    setState(() => _log += '$s\n');
  }

  Future<void> _runOffline() async {
    if (_running) return;

    setState(() {
      _running = true;
      _saving = false;
      _hasResult = false;
      _log = '';
      _vRawPlot.clear();
      _v10Plot.clear();
      _predPlot.clear();
      _truePlot.clear();
      _t = null;
      _vRaw = null;
      _v10Full = null;
      _trueSlm = null;
      _predSlm = null;
    });

    try {
      _appendLog(
          '=== Offline (Scheme-A) | model=$_currentModel | asset=$_currentAssetCsv ===');
      _appendLog('win=$winSize, stride=$stride, dropRatio=$dropRatio');
      _appendLog(
          'MATLAB-aligned: filtfilt(10Hz) + mean0 in Dart; Kotlin does /sig + bestLag + denorm');

      final csvText = await rootBundle.loadString(_currentAssetCsv);
      final rows = _parseCsv(csvText);
      if (rows.isEmpty) {
        _appendLog('ERROR: CSV empty');
        return;
      }

      final nAll = rows.length;
      final s0 = startIdx.clamp(0, nAll - 1);
      final s1 = endIdx.clamp(0, nAll - 1);
      final seg = rows.sublist(s0, s1 + 1);
      final n = seg.length;

      _appendLog('Parsed=$nAll, segment=[$s0..$s1], N=$n');

      final t = Float64List(n);
      final vRaw = Float64List(n);
      final flowV = Float64List(n);

      for (int i = 0; i < n; i++) {
        t[i] = seg[i][0];
        vRaw[i] = seg[i][1];
        flowV[i] = seg[i][2];
      }

      // ---- True flow: voltage -> flow -> filtfilt(10Hz) -> mean0 (MATLAB)
      final flowTrue = Float64List(n);
      for (int i = 0; i < n; i++) {
        flowTrue[i] = 212.5 * (((flowV[i] - V_ZERO) / VDD) - 0.1) - 10.0;
      }
      final flowLP = Butterworth2Filtfilt10Hz.filtfilt(flowTrue, pad: 300);
      _removeMeanInPlace(flowLP);

      // ---- Piezo v10: filtfilt(10Hz) -> mean0 (MATLAB)
      final v10LP = Butterworth2Filtfilt10Hz.filtfilt(vRaw, pad: 300);
      _removeMeanInPlace(v10LP);

      // ---- Overlap-add prediction (Kotlin does v/sig only; bestLag fixed inside Kotlin)
      final predSlm = await _inferencer.runOffline(v10LP);

      // MATLAB edge behavior: pred_sum/w_sum leaves zeros where no window contributed
      // Our inferencer returns NaN where wSum==0; convert NaN -> 0 before mean removal
      for (int i = 0; i < predSlm.length; i++) {
        if (predSlm[i].isNaN) predSlm[i] = 0.0;
      }

      // MATLAB: flow_pred = flow_pred - mean(flow_pred)
      _removeMeanInPlace(predSlm);

      // ---- Downsample for plots
      const int ds = 10;
      for (int i = 0; i < n; i += ds) {
        _vRawPlot.add(DataPoint(t[i], vRaw[i]));
        _v10Plot.add(DataPoint(t[i], v10LP[i]));
        _predPlot.add(DataPoint(t[i], predSlm[i]));
        _truePlot.add(DataPoint(t[i], flowLP[i]));
      }

      _t = t;
      _vRaw = vRaw;
      _v10Full = v10LP;
      _trueSlm = flowLP;
      _predSlm = predSlm;

      setState(() => _hasResult = true);
      _appendLog('=== Done (ready to SAVE) ===');
    } catch (e, st) {
      _appendLog('ERROR: $e');
      _appendLog('$st');
    } finally {
      if (mounted) setState(() => _running = false);
    }
  }

  Future<void> _saveCsv() async {
    if (_running || _saving || !_hasResult) return;

    final t = _t;
    final vRaw = _vRaw;
    final v10Full = _v10Full;
    final trueSlm = _trueSlm;
    final predSlm = _predSlm;

    if (t == null ||
        vRaw == null ||
        v10Full == null ||
        trueSlm == null ||
        predSlm == null) {
      _appendLog('ERROR: buffers missing');
      return;
    }

    setState(() => _saving = true);

    try {
      final ts = DateTime.now().toIso8601String().replaceAll(':', '-');

      // ✅ 這行一定要在同一個 scope 內宣告並使用
      final fileTag = _basenameNoExt(_currentAssetCsv);

      final path = '/data/user/0/com.example.real_time_plot_formal/app_flutter/'
          'offline_schemeA_${_currentModel}_${fileTag}_$ts.csv';

      final file = File(path);
      final sink = file.openWrite(mode: FileMode.writeOnly);

      // UTF-8 BOM (Excel friendly)
      sink.add(const [0xEF, 0xBB, 0xBF]);

      // 欄位名稱統一
      sink.write('t_s,vraw_v,v10_filtfilt_mean0_v,true_slm,pred_slm\r\n');

      String fmt(double x) => x.isNaN ? '' : x.toStringAsFixed(6);

      const int chunk = 5000;
      final sb = StringBuffer();
      final n = t.length;

      for (int i = 0; i < n; i++) {
        sb
          ..write(t[i].toStringAsFixed(6))
          ..write(',')
          ..write(vRaw[i].toStringAsFixed(6))
          ..write(',')
          ..write(fmt(v10Full[i]))
          ..write(',')
          ..write(fmt(trueSlm[i]))
          ..write(',')
          ..write(fmt(predSlm[i]))
          ..write('\r\n');

        if ((i + 1) % chunk == 0) {
          sink.write(sb.toString());
          sb.clear();
          await sink.flush();
          await Future.delayed(Duration.zero);
          _appendLog('[SAVE] ${(i + 1)}/$n');
        }
      }

      if (sb.isNotEmpty) sink.write(sb.toString());

      await sink.flush();
      await sink.close();

      _appendLog('Saved in app_flutter: ${path.split('/').last}');
    } catch (e, st) {
      _appendLog('SAVE ERROR: $e');
      _appendLog('$st');
    } finally {
      if (mounted) setState(() => _saving = false);
    }
  }

  List<List<double>> _parseCsv(String text) {
    final lines = text.split(RegExp(r'\r?\n'));
    final out = <List<double>>[];

    for (final line in lines) {
      final l = line.trim();
      if (l.isEmpty) continue;
      if (l.startsWith('#')) continue;

      final parts = l.split(',');
      if (parts.length < 3) continue;

      final a = double.tryParse(parts[0].trim());
      final b = double.tryParse(parts[1].trim());
      final c = double.tryParse(parts[2].trim());
      if (a == null || b == null || c == null) continue;

      out.add([a, b, c]);
    }
    return out;
  }

  void _removeMeanInPlace(Float64List x) {
    double sum = 0;
    for (int i = 0; i < x.length; i++) sum += x[i];
    final mean = sum / x.length.toDouble();
    for (int i = 0; i < x.length; i++) x[i] -= mean;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Offline Scheme-A (MATLAB-aligned)'),
        actions: [
          IconButton(
            onPressed: _running ? null : _runOffline,
            icon: const Icon(Icons.refresh),
          ),
        ],
      ),
      body: Column(
        children: [
          const SizedBox(height: 8),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 12),
            child: Row(
              children: [
                // Model selector
                DropdownButton<String>(
                  value: _currentModel,
                  items: _models
                      .map((m) => DropdownMenuItem(value: m, child: Text(m)))
                      .toList(),
                  onChanged: _running
                      ? null
                      : (v) async {
                          if (v == null) return;
                          await _initModel(v);
                          _appendLog('[MODEL] switched to $v');
                        },
                ),
                const SizedBox(width: 10),
                ElevatedButton(
                  onPressed: _running ? null : _runOffline,
                  child: Text(_running ? 'Running...' : 'Run Offline Test'),
                ),
                const SizedBox(width: 10),
                ElevatedButton(
                  onPressed:
                      (_running || _saving || !_hasResult) ? null : _saveCsv,
                  child: Text(_saving ? 'Saving...' : 'Save CSV'),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: Text(
                    _currentAssetCsv,
                    overflow: TextOverflow.ellipsis,
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 8),
          Expanded(
            child: ListView(
              padding: const EdgeInsets.all(12),
              children: [
                _buildVoltageChart(),
                const SizedBox(height: 16),
                _buildFlowChart(),
                const SizedBox(height: 16),
                _buildLogBox(),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildVoltageChart() {
    return SizedBox(
      height: 260,
      child: SfCartesianChart(
        title: const ChartTitle(text: 'Voltage (Raw vs v10 filtfilt+mean0)'),
        legend: const Legend(isVisible: true),
        primaryXAxis: const NumericAxis(title: AxisTitle(text: 'Time (s)')),
        primaryYAxis: const NumericAxis(title: AxisTitle(text: 'Voltage (V)')),
        series: <LineSeries<DataPoint, double>>[
          LineSeries<DataPoint, double>(
            name: 'Vraw',
            dataSource: _vRawPlot,
            xValueMapper: (p, _) => p.t,
            yValueMapper: (p, _) => p.y,
          ),
          LineSeries<DataPoint, double>(
            name: 'V10 (filtfilt+mean0)',
            dataSource: _v10Plot,
            xValueMapper: (p, _) => p.t,
            yValueMapper: (p, _) => p.y,
          ),
        ],
      ),
    );
  }

  Widget _buildFlowChart() {
    return SizedBox(
      height: 260,
      child: SfCartesianChart(
        title: ChartTitle(
            text: 'Flow (SLM) — Pred vs True | model=$_currentModel'),
        legend: const Legend(isVisible: true),
        primaryXAxis: const NumericAxis(title: AxisTitle(text: 'Time (s)')),
        primaryYAxis: const NumericAxis(title: AxisTitle(text: 'Flow (SLM)')),
        series: <LineSeries<DataPoint, double>>[
          LineSeries<DataPoint, double>(
            name: 'Pred SLM (phone, mean0)',
            dataSource: _predPlot,
            xValueMapper: (p, _) => p.t,
            yValueMapper: (p, _) => p.y,
          ),
          LineSeries<DataPoint, double>(
            name: 'True SLM (filtfilt+mean0)',
            dataSource: _truePlot,
            xValueMapper: (p, _) => p.t,
            yValueMapper: (p, _) => p.y,
          ),
        ],
      ),
    );
  }

  Widget _buildLogBox() {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        border: Border.all(),
        borderRadius: BorderRadius.circular(8),
      ),
      child: SelectableText(
        _log.isEmpty ? 'Log will appear here.' : _log,
        style: const TextStyle(fontFamily: 'monospace', fontSize: 12),
      ),
    );
  }
}
