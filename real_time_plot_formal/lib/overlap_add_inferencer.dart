// lib/overlap_add_inferencer.dart
import 'dart:math';
import 'dart:typed_data';

typedef InferFn = Future<Float32List> Function(Float32List window);

class OverlapAddInferencer {
  final int winSize;
  final int stride;
  final double dropRatio;
  final InferFn inferFn;

  late final Float64List _w; // taper weights

  OverlapAddInferencer({
    required this.inferFn,
    this.winSize = 2000,
    this.stride = 200,
    this.dropRatio = 0.05,
  }) {
    _w = _buildTaperWeights(winSize, (dropRatio * winSize).round());
  }

  Float64List _buildTaperWeights(int n, int dropEdge) {
    final w = Float64List(n);
    for (int i = 0; i < n; i++) w[i] = 1.0;

    if (dropEdge <= 0) return w;
    if (dropEdge == 1) return w;

    for (int i = 0; i < dropEdge; i++) {
      final v = i / (dropEdge - 1);
      w[i] = v;
      w[n - dropEdge + i] = 1.0 - v;
    }
    return w;
  }

  // ===========================
  // OFFLINE: full signal
  // ===========================
  Future<Float64List> runOffline(Float64List v10) async {
    final n = v10.length;
    final predSum = Float64List(n);
    final wSum = Float64List(n);

    for (int s = 0; s <= n - winSize; s += stride) {
      final win = Float32List(winSize);
      for (int i = 0; i < winSize; i++) {
        win[i] = v10[s + i].toDouble();
      }

      final y = await inferFn(win); // Kotlin returns SLM (after de-norm)
      final m = min(winSize, y.length);

      for (int i = 0; i < m; i++) {
        final idx = s + i;
        final ww = _w[i];
        predSum[idx] += y[i].toDouble() * ww;
        wSum[idx] += ww;
      }
    }

    for (int i = 0; i < n; i++) {
      predSum[i] = (wSum[i] > 0) ? (predSum[i] / wSum[i]) : double.nan;
    }
    return predSum;
  }

  // ===========================
  // REALTIME: streaming overlap-add
  // - You push v10 chunks
  // - It runs one inference every stride samples
  // - It overlap-adds window predictions into accumulators
  // - Returns current predicted series (same length as buffer)
  // ===========================
  final List<double> _v10Buf = [];
  final List<double> _predSum = [];
  final List<double> _wSum = [];
  int _consumed = 0;

  Future<List<double>> pushV10Samples(List<double> newV10) async {
    if (newV10.isEmpty) return const [];

    for (final x in newV10) {
      _v10Buf.add(x);
      _predSum.add(0.0);
      _wSum.add(0.0);
    }

    while (_v10Buf.length - _consumed >= stride) {
      _consumed += stride;
      final end = _consumed;
      final start = end - winSize;
      if (start < 0) continue;

      final win = Float32List(winSize);
      for (int i = 0; i < winSize; i++) {
        win[i] = _v10Buf[start + i].toDouble();
      }

      final y = await inferFn(win);
      final m = min(winSize, y.length);

      for (int i = 0; i < m; i++) {
        final idx = start + i;
        final ww = _w[i];
        _predSum[idx] += y[i].toDouble() * ww;
        _wSum[idx] += ww;
      }
    }

    final out = List<double>.filled(_v10Buf.length, double.nan);
    for (int i = 0; i < out.length; i++) {
      out[i] = (_wSum[i] > 0) ? (_predSum[i] / _wSum[i]) : double.nan;
    }
    return out;
  }

  void resetRealtime() {
    _v10Buf.clear();
    _predSum.clear();
    _wSum.clear();
    _consumed = 0;
  }
}
