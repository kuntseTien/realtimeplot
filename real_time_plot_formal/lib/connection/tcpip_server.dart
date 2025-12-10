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
    try {
      _server = await ServerSocket.bind(InternetAddress.anyIPv4, port);
      print(
          'TCP Server is running on ${_server!.address.address}:${_server!.port}');

      _server!.listen((Socket client) {
        print(
            'New client connected: ${client.remoteAddress.address}:${client.remotePort}');
        client.listen((data) {
          print('Data received from client: $data');
          var newData = onDataReceived(data);
          dataProcessor?.updateDataSource(newData);
        }, onDone: () {
          print(
              'Client disconnected: ${client.remoteAddress.address}:${client.remotePort}');
        }, onError: (error) {
          print('Error: $error');
        });
      });
    } catch (e) {
      print('Failed to start TCP server: $e');
    }
  }

  void stop() {
    try {
      _server?.close();
      _server = null;
      print('TCP Server stopped');
    } catch (e) {
      print('Error stopping the TCP server: $e');
    }
  }
}

Future<String> getLocalIpAddress() async {
  try {
    for (var interface
        in await NetworkInterface.list(type: InternetAddressType.IPv4)) {
      for (var address in interface.addresses) {
        if (!address.isLoopback) {
          return address.address;
        }
      }
    }
  } catch (e) {
    print('Failed to get local IP address: $e');
  }
  return 'No IP address found';
}

void main() async {
  TCPServer server = TCPServer(onDataReceived: (data) {
    print('Data received: $data');
    return data; // 假設這裡直接返回接收到的數據，您可以根據需要處理
  });

  await server.start();

  String ipAddress = await getLocalIpAddress();
  print('Local IP Address: $ipAddress');
}
