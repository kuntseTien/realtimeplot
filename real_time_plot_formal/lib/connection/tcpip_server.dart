import 'dart:io';
import 'dart:async';

import '../data_struct/data_processor.dart';

const int esp32DefaultPort = 11520;

class TCPServer {
  ServerSocket? _server;
  DataProcessor? dataProcessor;

  final int port;
  final Function(List<int>) onDataReceived;

  TCPServer({this.port = esp32DefaultPort, required this.onDataReceived});

  Future<void> start() async {
    _server = await ServerSocket.bind(InternetAddress.anyIPv4, port);
    print('TCP Server is running on ${_server!.address.address}:${_server!.port}');

    _server!.listen((Socket client) {
      client.listen((data) {
        var newData = onDataReceived(data);
        dataProcessor?.updateDataSource(newData);
      });
    });
  }

  void stop() {
    _server?.close();
  }
}
