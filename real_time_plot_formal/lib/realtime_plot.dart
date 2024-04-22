import 'dart:async';
import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:syncfusion_flutter_charts/charts.dart';

import 'data_struct/data_point.dart';
import 'data_struct/data_processor.dart';
import 'connection/tcpip_server.dart';

import 'dart:typed_data';

double tRecord = 0.0;

class RealTimePlotPage extends StatefulWidget {
  final int initialDataLength;

  const RealTimePlotPage({Key? key, this.initialDataLength = 100})
      : super(key: key);

  @override
  State<RealTimePlotPage> createState() => _RealTimePlotState();
}

class _RealTimePlotState extends State<RealTimePlotPage> {
  final List<DataPoint> _dataSource = [];
  final DataProcessor _dataProcessor = DataProcessor();
  late final StreamSubscription<List<DataPoint>> _dataSubscription;
  TCPServer? _tcpServer;
  ChartSeriesController? _chartSeriesController;

  @override
  void initState() {
    super.initState();
    _dataSubscription = _dataProcessor.dataStream.listen((newData) {
      updateRealTimePlot(newData);
    });

    // tcpip server
    _tcpServer = TCPServer(
      onDataReceived: onDataReceivedCallBack,
    );
    _tcpServer!.dataProcessor = _dataProcessor;
    _tcpServer!.start();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Real Time Plot'),
      ),
      body: SfCartesianChart(
        series: <CartesianSeries>[
          SplineSeries<DataPoint, double>(
            dataSource: _dataSource,
            xValueMapper: (DataPoint data, _) => data.t,
            yValueMapper: (DataPoint data, _) => data.y,
            onRendererCreated: (ChartSeriesController controller) {
              _chartSeriesController = controller; // 新增
            },
          )
        ],
      ),
    );
  }

  @override
  void dispose() {
    _dataSubscription.cancel();
    _dataProcessor.dispose();
    _tcpServer?.stop();
    super.dispose();
  }

  // void updateRealTimePlot(List<DataPoint> newDataPoints) {
  //   setState(() {
  //     print(_dataSource.length);
  //     int overflowLength = _dataSource.length + newDataPoints.length - widget.initialDataLength; // 假設固定長度為10
  //     if (overflowLength > 0) {
  //       _dataSource.removeRange(0, overflowLength);
  //     }
  //     _dataSource.addAll(newDataPoints);
  //   });
  // }

  void updateRealTimePlot(List<DataPoint> newDataPoints) {
    int overflowLength =
        _dataSource.length + newDataPoints.length - widget.initialDataLength;
    if (overflowLength > 0) {
      _dataSource.removeRange(0, overflowLength);
    }
    _dataSource.addAll(newDataPoints);

    // 更新圖表數據而不使用 setState
    _chartSeriesController?.updateDataSource(
      addedDataIndexes: List.generate(newDataPoints.length,
          (index) => _dataSource.length - newDataPoints.length + index),
      removedDataIndexes: overflowLength > 0
          ? List.generate(overflowLength, (index) => index)
          : [],
    );
  }
}

// functions
List<DataPoint> onDataReceivedCallBack(List<int> tcpByteData) {
  // var dataString = utf8.decode(tcpByteData);
  // List<DataPoint> dataPoints = parseData(dataString);
  List<DataPoint> dataPoints = parseFloatArray(tcpByteData);
  return dataPoints;
}

List<DataPoint> parseFloatArray(List<int> floatArrayByteData) {
  // const int requiredBytes = 8000;
  bool can_be_devided_by_4 = (floatArrayByteData.length % 4 == 0);
  if (!can_be_devided_by_4) {
    throw Exception("Data length mod 4 should be 0!");
  }

  var u8data = Uint8List.fromList(floatArrayByteData);
  Float32List floatArrayData = u8data.buffer.asFloat32List();
  List<DataPoint> dataPoints = [];

  for (int i = 0; i < floatArrayByteData.length / 4; i++) {
    double y = floatArrayData[i]; // 读取每个浮点数
    // TODO: 新增 class 以去除 tRecord 作為全域變數
    dataPoints.add(DataPoint(tRecord, y));
    tRecord += 0.0005; // 更新时间
  }

  return dataPoints;
}

List<DataPoint> parseData(String dataString) {
  List<DataPoint> dataPoints = [];
  var lines = dataString.split(';');
  for (var line in lines) {
    var parts = line.split(',');
    if (parts.length == 2) {
      double t = double.parse(parts[0]);
      double y = double.parse(parts[1]);
      dataPoints.add(DataPoint(t, y));
    }
  }
  return dataPoints;
}
