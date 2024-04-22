import 'dart:async';

import 'data_point.dart';

class DataProcessor {
  final StreamController<List<DataPoint>> _dataStreamController = StreamController.broadcast();

  Stream<List<DataPoint>> get dataStream => _dataStreamController.stream;

  void updateDataSource(List<DataPoint> newData) {
    _dataStreamController.add(newData);
  }

  void dispose() {
    _dataStreamController.close();
  }
}