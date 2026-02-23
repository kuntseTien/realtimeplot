import 'dart:async';
import 'dart:io';
import 'dart:math';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:syncfusion_flutter_charts/charts.dart';

import 'onnx/onnx_stand_channel.dart';

class DataPoint {
  final double t;
  final double y;
  DataPoint(this.t, this.y);
}

// ======================================================
// ✅ Realtime causal Butterworth (SOS) 10 Hz LP
// (跟你 realtime 那份一樣：兩段 SOS + gains + states)
// ======================================================
class ButterworthFilter10Hz {
  final List<List<double>> sos = const [
    [1, 2, 1, -1.975269634851873, 0.97624479235944],
    [1, 2, 1, -1.9426382305401135, 0.9435972784703671],
  ];

  final List<double> gains = const [
    0.00024378937689168925,
    0.00023976198256338974,
  ];

  final List<List<double>> states = List.generate(2, (_) => [0.0, 0.0]);

  double apply(double input) {
    double x = input;
    for (int i = 0; i < sos.length; i++) {
      final b = sos[i];
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

class OfflineTestPage extends StatefulWidget {
  const OfflineTestPage({super.key});

  @override
  State<OfflineTestPage> createState() => _OfflineTestPageState();
}

class _OfflineTestPageState extends State<OfflineTestPage> {
  // ---- Config ----
  static const String kAssetCsv = 'assets/testdata/chirustand03.csv';
  static const int fs = 1000;
  static const int startIdx = 4000;
  static const int endIdx = 74000;

  // Flow voltage -> SLM formula (同你 MATLAB)
  static const double VDD = 5.0;
  static const double V_ZERO = 0.74;

  // sliding window
  static const int winSize = 2000;
  static const int stride = 200;

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
  Float64List? _v10;
  Float64List? _trueSlm;
  Float64List? _predSlm;

  @override
  void initState() {
    super.initState();
    // 只初始化一次（STAND）
    OnnxStandChannel.init("STAND");
  }

  void _appendLog(String s) {
    // ignore: avoid_print
    print(s);
    if (!mounted) return;
    setState(() => _log += '$s\n');
  }

  // ===========================
  // Main: Run offline pipeline
  // ===========================
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
      _v10 = null;
      _trueSlm = null;
      _predSlm = null;
    });

    try {
      _appendLog('=== Offline CSV Test ===');
      _appendLog('Loading $kAssetCsv ...');

      final csvText = await rootBundle.loadString(kAssetCsv);
      final rows = _parseCsv(csvText);
      if (rows.isEmpty) {
        _appendLog('❌ CSV empty.');
        return;
      }

      final nAll = rows.length;
      _appendLog('Parsed samples = $nAll');

      final s0 = startIdx.clamp(0, nAll - 1);
      final s1 = endIdx.clamp(0, nAll - 1);
      final seg = rows.sublist(s0, s1 + 1);
      final n = seg.length;
      _appendLog('Using segment: [$s0 .. $s1] => N=$n');

      // allocate
      final t = Float64List(n);
      final vRaw = Float64List(n);
      final flowV = Float64List(n);

      for (int i = 0; i < n; i++) {
        t[i] = seg[i][0];
        vRaw[i] = seg[i][1];
        flowV[i] = seg[i][2];
      }

      // ---- v10 (causal butterworth 10Hz) then mean removal ----
      final v10 = _causalButterworth10Hz(vRaw);
      _removeMeanInPlace(v10);

      // ---- true flow (col3 formula) then causal butterworth 10Hz then mean removal ----
      final flowTrue = Float64List(n);
      for (int i = 0; i < n; i++) {
        flowTrue[i] = 212.5 * (((flowV[i] - V_ZERO) / VDD) - 0.1) - 10.0;
      }
      final true10 = _causalButterworth10Hz(flowTrue);
      _removeMeanInPlace(true10);

      // log v10 stats
      double inMin = 1e18, inMax = -1e18, inMean = 0;
      for (int i = 0; i < n; i++) {
        final x = v10[i];
        inMin = min(inMin, x);
        inMax = max(inMax, x);
        inMean += x;
      }
      inMean /= n;
      _appendLog(
          'v10 input stats: min=${inMin.toStringAsFixed(6)} max=${inMax.toStringAsFixed(6)} mean=${inMean.toStringAsFixed(6)}');

      // ---- overlap-add inference ----
      final predSlm = Float64List(n);
      final wSum = Float64List(n);

      for (int s = 0; s <= n - winSize; s += stride) {
        final win = Float32List(winSize);
        for (int i = 0; i < winSize; i++) {
          win[i] = v10[s + i]; // double -> float32 (Dart 允許)
        }

        final y =
            await OnnxStandChannel.infer(win); // Kotlin 回來 SLM (Float32List)

        for (int i = 0; i < winSize && i < y.length; i++) {
          final idx = s + i;
          predSlm[idx] += y[i].toDouble();
          wSum[idx] += 1.0;
        }

        if (s % (stride * 50) == 0) {
          _appendLog('Infer progress: s=$s / ${n - winSize}');
          await Future.delayed(Duration.zero);
        }
      }

      for (int i = 0; i < n; i++) {
        predSlm[i] = (wSum[i] > 0) ? (predSlm[i] / wSum[i]) : 0.0;
      }

      // ---- Pred stats ----
      int zeroCount = 0;
      int nanCount = 0;
      double predMin = 1e18, predMax = -1e18, predMean = 0;
      int valid = 0;

      for (int i = 0; i < n; i++) {
        final x = predSlm[i];
        if (x.isNaN) {
          nanCount++;
          continue;
        }
        if (x == 0.0) {
          zeroCount++;
          continue;
        }
        predMin = min(predMin, x);
        predMax = max(predMax, x);
        predMean += x;
        valid++;
      }
      predMean = (valid > 0) ? (predMean / valid) : 0.0;

      _appendLog(
          'Pred SLM valid=$valid, zeroCount=$zeroCount, nanCount=$nanCount');
      _appendLog(
          'Pred SLM min/max (valid only) = [${predMin.toStringAsFixed(6)}, ${predMax.toStringAsFixed(6)}], mean=${predMean.toStringAsFixed(6)}');

      // ---- Build plots (downsample) ----
      const int ds = 10;
      for (int i = 0; i < n; i += ds) {
        final tt = t[i];
        _vRawPlot.add(DataPoint(tt, vRaw[i]));
        _v10Plot.add(DataPoint(tt, v10[i]));
        _predPlot.add(DataPoint(tt, predSlm[i]));
        _truePlot.add(DataPoint(tt, true10[i]));
      }

      _appendLog('Preview (every 100 samples, first 5s):');
      final maxShow = min(n, 5000);
      for (int i = 0; i < maxShow; i += 100) {
        _appendLog(
            't=${t[i].toStringAsFixed(3)}  predSLM=${predSlm[i].toStringAsFixed(3)}  trueSLM=${true10[i].toStringAsFixed(3)}');
      }

      // store for save
      _t = t;
      _vRaw = vRaw;
      _v10 = v10;
      _trueSlm = true10;
      _predSlm = predSlm;

      setState(() {
        _hasResult = true;
      });
      _appendLog('=== Done (ready to SAVE) ===');
    } catch (e, st) {
      _appendLog('❌ ERROR: $e');
      _appendLog('$st');
    } finally {
      if (mounted) setState(() => _running = false);
    }
  }

  // ===========================
  // SAVE CSV (chunk writing + CRLF + BOM)
  // ===========================
  Future<void> _saveCsv() async {
    _appendLog(
        '[SAVE] pressed. running=$_running saving=$_saving hasResult=$_hasResult');
    if (_running || _saving) return;
    if (!_hasResult) {
      _appendLog('❌ No result to save yet.');
      return;
    }
    final t = _t;
    final vRaw = _vRaw;
    final v10 = _v10;
    final trueSlm = _trueSlm;
    final predSlm = _predSlm;
    if (t == null ||
        vRaw == null ||
        v10 == null ||
        trueSlm == null ||
        predSlm == null) {
      _appendLog('❌ Internal: result buffers missing.');
      return;
    }

    setState(() => _saving = true);

    try {
      final ts = DateTime.now().toIso8601String().replaceAll(':', '-');
      final path =
          '/data/user/0/com.example.real_time_plot_formal/app_flutter/offline_pred_true_$ts.csv';
      _appendLog('[SAVE] writing to: $path');

      final file = File(path);
      final sink = file.openWrite(mode: FileMode.writeOnly);

      // BOM for Excel
      sink.add(const [0xEF, 0xBB, 0xBF]);

      // Header (CRLF)
      sink.write('t_s,vraw_v,v10_v,true_slm,pred_slm\r\n');

      const int chunk = 5000;
      final sb = StringBuffer();
      final n = t.length;

      for (int i = 0; i < n; i++) {
        sb
          ..write(t[i].toStringAsFixed(6))
          ..write(',')
          ..write(vRaw[i].toStringAsFixed(6))
          ..write(',')
          ..write(v10[i].toStringAsFixed(6))
          ..write(',')
          ..write(trueSlm[i].toStringAsFixed(6))
          ..write(',')
          ..write(predSlm[i].toStringAsFixed(6))
          ..write('\r\n');

        if ((i + 1) % chunk == 0) {
          sink.write(sb.toString());
          sb.clear();
          await sink.flush();
          await Future.delayed(Duration.zero); // 讓 UI 不會卡死
          _appendLog('[SAVE] progress ${(i + 1)}/$n');
        }
      }

      if (sb.isNotEmpty) {
        sink.write(sb.toString());
        sb.clear();
      }

      await sink.flush();
      await sink.close();

      _appendLog('✅ Exported CSV: $path');
      _appendLog(
          '👉 adb exec-out run-as com.example.real_time_plot_formal cat app_flutter/${path.split('/').last} > "C:\\Users\\kuntse\\Desktop\\Download_backu\\${path.split('/').last}"');
    } catch (e, st) {
      _appendLog('❌ SAVE ERROR: $e');
      _appendLog('$st');
    } finally {
      if (mounted) setState(() => _saving = false);
    }
  }

  // ===========================
  // CSV parse
  // ===========================
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

  // ===========================
  // Causal Butterworth 10Hz (realtime style)
  // ===========================
  Float64List _causalButterworth10Hz(Float64List x) {
    final y = Float64List(x.length);
    final f = ButterworthFilter10Hz();
    for (int i = 0; i < x.length; i++) {
      y[i] = f.apply(x[i]);
    }
    return y;
  }

  void _removeMeanInPlace(Float64List x) {
    double sum = 0;
    for (int i = 0; i < x.length; i++) sum += x[i];
    final mean = sum / x.length;
    for (int i = 0; i < x.length; i++) x[i] -= mean;
  }

  // ===========================
  // UI
  // ===========================
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Offline CSV → Voltage & Flow Plot'),
        actions: [
          IconButton(
            onPressed: _running ? null : _runOffline,
            icon: const Icon(Icons.refresh),
          )
        ],
      ),
      body: Column(
        children: [
          const SizedBox(height: 8),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 12),
            child: Row(
              children: [
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
                    kAssetCsv,
                    overflow: TextOverflow.ellipsis,
                  ),
                )
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
        title: const ChartTitle(
            text: 'Voltage (Raw & 10Hz Causal Butterworth Mean-Removed)'),
        legend: const Legend(isVisible: true),
        primaryXAxis: const NumericAxis(title: AxisTitle(text: 'Time (s)')),
        primaryYAxis: const NumericAxis(title: AxisTitle(text: 'Voltage (V)')),
        series: <LineSeries<DataPoint, double>>[
          LineSeries<DataPoint, double>(
            name: 'Vraw (col2)',
            dataSource: _vRawPlot,
            xValueMapper: (p, _) => p.t,
            yValueMapper: (p, _) => p.y,
          ),
          LineSeries<DataPoint, double>(
            name: 'V10 (Butterworth+mean0)',
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
        title: const ChartTitle(text: 'Flow (SLM) — Pred vs True'),
        legend: const Legend(isVisible: true),
        primaryXAxis: const NumericAxis(title: AxisTitle(text: 'Time (s)')),
        primaryYAxis: const NumericAxis(title: AxisTitle(text: 'Flow (SLM)')),
        series: <LineSeries<DataPoint, double>>[
          LineSeries<DataPoint, double>(
            name: 'Pred SLM (ONNX)',
            dataSource: _predPlot,
            xValueMapper: (p, _) => p.t,
            yValueMapper: (p, _) => p.y,
          ),
          LineSeries<DataPoint, double>(
            name: 'True SLM (col3 → formula → Butterworth+mean0)',
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
