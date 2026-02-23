package com.example.real_time_plot_formal

import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel

class MainActivity : FlutterActivity() {

    private lateinit var infer: OnnxModelInfer
    private val CHANNEL = "onnx_stand_channel"

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)

        infer = OnnxModelInfer(this)

        MethodChannel(
            flutterEngine.dartExecutor.binaryMessenger,
            CHANNEL
        ).setMethodCallHandler { call, result ->

            when (call.method) {

                "init" -> {
                    val modelName = call.argument<String>("model") ?: "STAND"
                    try {
                        infer.init(modelName)
                        result.success(null)
                    } catch (e: Exception) {
                        result.error("INIT_FAIL", e.message, null)
                    }
                }

                "infer" -> {
                    try {
                        val input = call.argument<List<Double>>("input") ?: emptyList()
                        val floatIn = FloatArray(input.size) { i -> input[i].toFloat() }

                        val output = infer.infer(floatIn)
                        result.success(output.toList())
                    } catch (e: Exception) {
                        result.error("INFER_FAIL", e.message, null)
                    }
                }

                else -> result.notImplemented()
            }
        }
    }
}
