import 'dart:async';
import 'dart:io';
import 'data_point.dart';

class DataProcessor {
  final StreamController<List<DataPoint>> _dataStreamController =
      StreamController.broadcast();

  Stream<List<DataPoint>> get dataStream => _dataStreamController.stream;

  void updateDataSource(List<DataPoint> newData) {
    _dataStreamController.add(newData);
  }

  void dispose() {
    _dataStreamController.close();
  }
}

void saveDataToCSV(List<DataPoint> dataPoints, String filePath) {
  List<String> lines = dataPoints.map((data) => "${data.t},${data.y}").toList();
  String csvData = "time,value\n" + lines.join('\n');
  File file = File(filePath);
  file.writeAsStringSync(csvData);
}
