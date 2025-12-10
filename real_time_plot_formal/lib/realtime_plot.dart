import 'dart:async';
import 'dart:typed_data';
import 'dart:io';
import 'dart:isolate';
import 'dart:math';

import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';
import 'package:syncfusion_flutter_charts/charts.dart';

import 'data_struct/data_point.dart'; // 確保此處有正確的導入路徑
import 'data_struct/data_processor.dart';
import 'connection/tcpip_server.dart';

double tRecord = 0.0;
double peakValue = double.negativeInfinity;
File? logFile;

class ButterworthFilter {
  List<List<double>> sosCoefficients = [
    [
      1,
      2,
      1,
      -1.975269634851873012948431096447166055441,
      0.976244792359439950146793307794723659754
    ],
    [
      1,
      2,
      1,
      -1.942638230540113530864232416206505149603,
      0.943597278470367117897410480509279295802
    ]
  ];
  List<double> gains = [
    0.000243789376891689250990286064180168069,
    0.000239761982563389738085449232052326352
  ];
  List<List<double>> states = List.generate(2, (_) => [0.0, 0.0]);

  double apply(double input) {
    double x = input;
    for (int i = 0; sosCoefficients.length > i; i++) {
      List<double> b = [
        sosCoefficients[i][0],
        sosCoefficients[i][1],
        sosCoefficients[i][2]
      ];
      List<double> a = [1.0, sosCoefficients[i][3], sosCoefficients[i][4]];
      List<double> state = states[i];
      double gain = gains[i];

      double v = x - a[1] * state[0] - a[2] * state[1];
      double y = gain * (b[0] * v + b[1] * state[0] + b[2] * state[1]);

      state[1] = state[0];
      state[0] = v;

      x = y;
    }
    return x;
  }
}

class RealTimePlotPage extends StatefulWidget {
  final int initialDataLength;

  const RealTimePlotPage({Key? key, this.initialDataLength = 100})
      : super(key: key);

  @override
  State<RealTimePlotPage> createState() => _RealTimePlotState();
}

class _RealTimePlotState extends State<RealTimePlotPage> {
  final List<DataPoint> _dataSource = [];
  final List<DataPoint> _allDataPoints = [];
  final DataProcessor _dataProcessor = DataProcessor();
  late final StreamSubscription<List<DataPoint>> _dataSubscription;
  TCPServer? _tcpServer;
  ChartSeriesController? _chartSeriesController;
  Isolate? logIsolate;
  SendPort? logSendPort;
  ButterworthFilter filter = ButterworthFilter();
  double xAxisMin = 0;
  double xAxisMax = 10; // 改為10秒
  int bufferSize = 2000; // 你的buffer size

  @override
  void initState() {
    super.initState();
    initializeLogFile();
    startLogIsolate();
    _dataSubscription = _dataProcessor.dataStream.listen((newData) {
      updateRealTimePlot(newData);
    });

    _tcpServer = TCPServer(onDataReceived: onDataReceivedCallBack);
    _tcpServer!.dataProcessor = _dataProcessor;
    _tcpServer!.start();
  }

  void startLogIsolate() async {
    final receivePort = ReceivePort();
    logIsolate = await Isolate.spawn(_logDataHandler, receivePort.sendPort);
    receivePort.listen((data) {
      if (data is SendPort) {
        logSendPort = data;
      }
    });
  }

  static void _logDataHandler(SendPort sendPort) {
    final port = ReceivePort();
    sendPort.send(port.sendPort);
    port.listen((message) {
      if (message is String) {
        File(logFile!.path)
            .writeAsString(message, mode: FileMode.append, flush: true);
      }
    });
  }

  void logData(List<DataPoint> dataPoints) {
    String dataString =
        dataPoints.map((dp) => '${dp.t},${dp.y},${dp.originalY}').join('\n') +
            '\n'; // 現在也記錄原始數據
    logSendPort?.send(dataString);
  }

  Future<void> initializeLogFile() async {
    final directory = await getExternalStorageDirectory();
    if (directory != null) {
      String filePath = '${directory.path}/dataPoints.csv';
      logFile = File(filePath);
    } else {
      print("Unable to get external storage directory");
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('realtimeplot'),
        actions: [
          IconButton(
            icon: const Icon(Icons.info_outline),
            onPressed: () => showDialog(
              context: context,
              builder: (context) => AlertDialog(
                title: const Text("峰值"),
                content: Text("當前峰值：$peakValue"),
                actions: <Widget>[
                  TextButton(
                    onPressed: () => Navigator.of(context).pop(),
                    child: const Text('關閉'),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
      body: SfCartesianChart(
        primaryXAxis: NumericAxis(
          minimum: xAxisMin, // 設置X軸最小值
          maximum: xAxisMax, // 設置X軸最大值
          interval: 1,
        ),
        primaryYAxis: NumericAxis(
          minimum: 0, // 設置Y軸最小值
          maximum: 5, // 設置Y軸最大值
          interval: 1,
        ),
        series: <CartesianSeries>[
          SplineSeries<DataPoint, double>(
            dataSource: _dataSource,
            xValueMapper: (DataPoint data, _) => data.t,
            yValueMapper: (DataPoint data, _) => data.y,
            onRendererCreated: (ChartSeriesController controller) {
              _chartSeriesController = controller;
            },
          )
        ],
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: () {
          if (logFile != null) {
            saveDataToCSV(_allDataPoints, logFile!.path);
          } else {
            print("Log file is not initialized.");
          }
        },
        tooltip: 'Save Data to CSV',
        child: const Icon(Icons.save),
      ),
    );
  }

  void saveDataToCSV(List<DataPoint> data, String filePath) {
    final String csvData =
        data.map((dp) => '${dp.t},${dp.originalY},${dp.y}').join('\n') +
            '\n'; // 現在 CSV 包含原始數據和過濾數據
    File(filePath).writeAsString(csvData).catchError((e) {
      print("Error saving data: $e");
    });
  }

  @override
  void dispose() {
    _dataSubscription.cancel();
    _dataProcessor.dispose();
    _tcpServer?.stop();
    logIsolate?.kill(priority: Isolate.immediate);
    super.dispose();
  }

  void updateRealTimePlot(List<DataPoint> newDataPoints) {
    // 將新數據加入到所有數據點的列表中
    _allDataPoints.addAll(newDataPoints);
    logData(newDataPoints);

    // 分批次添加數據點，避免一次性添加大量數據
    int batchSize = 200; // 每次更新100個數據點
    for (int i = 0; i < newDataPoints.length; i += batchSize) {
      int end = min(i + batchSize, newDataPoints.length);
      List<DataPoint> batch = newDataPoints.sublist(i, end);

      setState(() {
        _dataSource.addAll(batch);
        // 移除超過10秒範圍的數據點
        _dataSource.removeWhere((point) => point.t < tRecord - 10);
        // 更新X軸範圍
        xAxisMin = max(0, tRecord - 10);
        xAxisMax = tRecord;
      });

      _chartSeriesController?.updateDataSource(
        addedDataIndexes: List.generate(
            batch.length, (index) => _dataSource.length - batch.length + index),
        removedDataIndexes: [],
      );
    }

    double newPeak = newDataPoints.fold<double>(
        peakValue, (prev, element) => element.y > prev ? element.y : prev);
    if (newPeak != peakValue) {
      setState(() {
        peakValue = newPeak;
      });
      print("新峰值：$peakValue");
    }
  }

  List<DataPoint> onDataReceivedCallBack(List<int> tcpByteData) {
    return parseFloatArray(tcpByteData);
  }

  List<DataPoint> parseFloatArray(List<int> floatArrayByteData) {
    if (floatArrayByteData.length % 4 != 0) {
      throw Exception("數據長度必需是4的倍數！");
    }
    var u8data = Uint8List.fromList(floatArrayByteData);
    Float32List floatArrayData = u8data.buffer.asFloat32List();
    List<DataPoint> dataPoints = [];

    // 減少數據點的頻率：每隔一個數據點取一次
    for (int i = 0; i < floatArrayData.length; i += 2) {
      double originalY = floatArrayData[i];
      double filteredY = filter.apply(originalY); // 應用濾波器
      dataPoints
          .add(DataPoint(tRecord, filteredY, originalY)); // 現在也傳遞 originalY
      tRecord += 0.001; // 調整時間間隔
    }

    return dataPoints;
  }
}
