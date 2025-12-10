// import 'package:butter/butter.dart';

// class ButterworthFilter {
//   late Butter _butter;

//   ButterworthFilter() {
//     // 初始化濾波器，參數為：取樣率 2000Hz，截止頻率 10Hz，階數 4
//     _butter = Butter.lowPass(4, 2000, 10);
//   }

//   List<double> filterData(Float32List inputData) {
//     List<double> outputData = List<double>.filled(inputData.length, 0.0);
//     for (int i = 0; i < inputData.length; i++) {
//       outputData[i] = _butter.filter(inputData[i]);
//     }
//     return outputData;
//   }
// }
