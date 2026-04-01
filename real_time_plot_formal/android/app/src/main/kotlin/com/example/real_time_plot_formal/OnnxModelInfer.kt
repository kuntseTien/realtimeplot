package com.example.real_time_plot_formal

import android.content.Context
import android.util.Log
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import java.nio.FloatBuffer
import java.security.MessageDigest
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

class OnnxModelInfer(private val context: Context) {

    companion object {
        private const val TAG = "ONNX_MODEL"
        const val WINDOW_SIZE = 2000
    }

    data class ModelConfig(
        val assetName: String,
        val v10Sig: Float,
        val gfMin: Float,
        val gfMax: Float,
        val v10MeanAll: Float,   // 保留欄位（Scheme-A infer 不使用）
        val bestLag: Int
    )

    private val modelMap = mapOf(
        "STAND" to ModelConfig(
            assetName = "STAND.onnx",
            v10Sig = 0.203235355f,
            gfMin = -10.30820446f,
            gfMax = 10.89035348f,
            v10MeanAll = 1.247518661f, // ignored in Scheme-A
            bestLag = 0
        ),
        "DB" to ModelConfig(
            assetName = "DB.onnx",
            v10Sig = 0.118292885f,
            gfMin = -10.53231848f,
            gfMax = 12.27076919f,
            v10MeanAll = 1.250816579f, // ignored in Scheme-A
            bestLag = 0
        ),
        "SLEEP" to ModelConfig(
            assetName = "SLEEP.onnx",
            v10Sig = 0.082248704f,
            gfMin = -4.55798688f,
            gfMax = 3.616109183f,
            v10MeanAll = 1.238965648f, // ignored in Scheme-A
            bestLag = 0
        )
    )

    private lateinit var env: OrtEnvironment
    private lateinit var session: OrtSession
    private lateinit var currentModel: ModelConfig
    private var initialized = false

    private var inputName: String = ""
    private var outputName: String = ""

    fun init(modelName: String) {
        val model = modelMap[modelName]
            ?: throw IllegalArgumentException("Unknown model: $modelName")

        if (initialized) {
            if (::currentModel.isInitialized && currentModel.assetName == model.assetName) {
                Log.i(TAG, "Already initialized: $modelName")
                return
            }
            try {
                session.close()
                Log.i(TAG, "Previous session closed")
            } catch (e: Exception) {
                Log.w(TAG, "Session close failed", e)
            }
            initialized = false
        }

        currentModel = model

        val bytes = context.assets
            .open("flutter_assets/assets/${model.assetName}")
            .use { it.readBytes() }

        Log.i(TAG, "Loading ${model.assetName}, size=${bytes.size}, md5=${md5(bytes)}")

        env = OrtEnvironment.getEnvironment()
        val opts = OrtSession.SessionOptions().apply {
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }

        session = env.createSession(bytes, opts)
        initialized = true

        inputName = session.inputNames.first()
        outputName = session.outputNames.first()

        Log.i(TAG, "ONNX init success: $modelName")
        Log.i(
            TAG,
            "Params: v10Sig=${currentModel.v10Sig}, gfMin=${currentModel.gfMin}, gfMax=${currentModel.gfMax}, " +
                    "v10MeanAll=${currentModel.v10MeanAll} (IGNORED in Scheme-A), bestLag=${currentModel.bestLag}"
        )
        Log.i(TAG, "InputNames=${session.inputNames}, OutputNames=${session.outputNames}")

        try {
            val inInfo = session.inputInfo[inputName]
            val outInfo = session.outputInfo[outputName]
            Log.i(TAG, "InputInfo($inputName) = $inInfo")
            Log.i(TAG, "OutputInfo($outputName) = $outInfo")

            val inTensorInfo = inInfo?.info as? TensorInfo
            val outTensorInfo = outInfo?.info as? TensorInfo
            Log.i(
                TAG,
                "InputTensor shape=${inTensorInfo?.shape?.contentToString()} dimNames=${inTensorInfo?.dimensionNames}"
            )
            Log.i(
                TAG,
                "OutputTensor shape=${outTensorInfo?.shape?.contentToString()} dimNames=${outTensorInfo?.dimensionNames}"
            )
        } catch (e: Exception) {
            Log.w(TAG, "Failed to print model IO shape", e)
        }
    }

    fun infer(input: FloatArray): FloatArray {
        require(initialized) { "ONNX not initialized" }
        require(input.size == WINDOW_SIZE) { "Input size must be $WINDOW_SIZE" }

        // =========================================================
        // Original Scheme-A
        // Flutter/Dart 已做:
        //   - 濾波
        //   - mean0
        // Kotlin 這裡再做:
        //   - / v10Sig
        // =========================================================
        val sig = currentModel.v10Sig
        val inputNorm = FloatArray(WINDOW_SIZE)

        var rawMin = Float.POSITIVE_INFINITY
        var rawMax = Float.NEGATIVE_INFINITY
        var inMin = Float.POSITIVE_INFINITY
        var inMax = Float.NEGATIVE_INFINITY
        var meanIn = 0f
        var sumSq = 0f

        for (i in 0 until WINDOW_SIZE) {
            val r = input[i]
            rawMin = min(rawMin, r)
            rawMax = max(rawMax, r)

            val v = r / sig
            inputNorm[i] = v

            meanIn += v
            sumSq += v * v
            inMin = min(inMin, v)
            inMax = max(inMax, v)
        }

        meanIn /= WINDOW_SIZE.toFloat()
        val varIn = (sumSq / WINDOW_SIZE.toFloat()) - meanIn * meanIn
        val stdIn = sqrt(max(varIn, 0f))

        Log.d(TAG, "Input raw range=[$rawMin, $rawMax] (mean0 expected)")
        Log.d(TAG, "Input norm mean=$meanIn, std=$stdIn, range=[$inMin, $inMax], v10Sig=$sig")

        val inputShape = longArrayOf(WINDOW_SIZE.toLong(), 1L, 1L)
        val fb = FloatBuffer.wrap(inputNorm)

        val outputSlm = FloatArray(WINDOW_SIZE)

        var yMin = Float.POSITIVE_INFINITY
        var yMax = Float.NEGATIVE_INFINITY

        OnnxTensor.createTensor(env, fb, inputShape).use { inputTensor ->
            session.run(mapOf(inputName to inputTensor)).use { result ->
                val outTensor = result[0] as OnnxTensor
                val rawBuf = outTensor.floatBuffer.duplicate()
                rawBuf.rewind()

                val n = rawBuf.remaining()
                Log.d(TAG, "Output floatCount=$n, outTensorInfo=${outTensor.info}")

                if (n < WINDOW_SIZE) {
                    throw IllegalStateException("Model output has only $n floats (< $WINDOW_SIZE).")
                }

                val tmp = FloatArray(n)
                rawBuf.get(tmp)

                val start = if (n == WINDOW_SIZE) 0 else (n - WINDOW_SIZE)

                val gfMin = currentModel.gfMin
                val gfMax = currentModel.gfMax
                val span = gfMax - gfMin

                // Apply fixed bestLag (currently 0)
                val lag = currentModel.bestLag

                for (i in 0 until WINDOW_SIZE) {
                    val src = i - lag
                    val j = ((src % WINDOW_SIZE) + WINDOW_SIZE) % WINDOW_SIZE
                    val y = tmp[start + j]

                    yMin = min(yMin, y)
                    yMax = max(yMax, y)

                    outputSlm[i] = (y + 1f) * 0.5f * span + gfMin
                }
            }
        }

        var outMin = Float.POSITIVE_INFINITY
        var outMax = Float.NEGATIVE_INFINITY
        var outMean = 0f
        for (i in 0 until WINDOW_SIZE) {
            val v = outputSlm[i]
            outMin = min(outMin, v)
            outMax = max(outMax, v)
            outMean += v
        }
        outMean /= WINDOW_SIZE.toFloat()

        Log.d(
            TAG,
            "Model y(range after shift)=[$yMin, $yMax] -> Output SLM range=[$outMin, $outMax], mean=$outMean"
        )

        return outputSlm
    }

    private fun md5(data: ByteArray): String {
        val md = MessageDigest.getInstance("MD5")
        return md.digest(data).joinToString("") { "%02x".format(it) }
    }
}