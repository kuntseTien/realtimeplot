package com.example.real_time_plot_formal

import android.content.Context
import android.util.Log
import ai.onnxruntime.*
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
        val gfMax: Float
    )

    private val modelMap = mapOf(
        "STAND" to ModelConfig("STAND.onnx", 0.203235f, -10.3082f, 10.8904f),
        "DB"    to ModelConfig("DB.onnx",    0.118293f, -10.5323f, 12.2708f),
        "SLEEP" to ModelConfig("SLEEP.onnx", 0.082249f,  -4.55799f, 3.61611f)
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
        Log.i(TAG, "Params: v10Sig=${currentModel.v10Sig}, gfMin=${currentModel.gfMin}, gfMax=${currentModel.gfMax}")
        Log.i(TAG, "InputNames=${session.inputNames}, OutputNames=${session.outputNames}")

        // 印出模型 IO 資訊（含 dimNames）
        try {
            val inInfo = session.inputInfo[inputName]
            val outInfo = session.outputInfo[outputName]
            Log.i(TAG, "InputInfo($inputName) = $inInfo")
            Log.i(TAG, "OutputInfo($outputName) = $outInfo")

            val inTensorInfo = (inInfo?.info as? TensorInfo)
            val outTensorInfo = (outInfo?.info as? TensorInfo)
            Log.i(TAG, "InputTensor shape=${inTensorInfo?.shape?.contentToString()} dimNames=${inTensorInfo?.dimensionNames}")
            Log.i(TAG, "OutputTensor shape=${outTensorInfo?.shape?.contentToString()} dimNames=${outTensorInfo?.dimensionNames}")
        } catch (e: Exception) {
            Log.w(TAG, "Failed to print model IO shape", e)
        }
    }

    fun infer(input: FloatArray): FloatArray {
        require(initialized) { "ONNX not initialized" }
        require(input.size == WINDOW_SIZE) { "Input size must be $WINDOW_SIZE" }

        // 0) window mean removal
        var meanRaw = 0f
        for (i in 0 until WINDOW_SIZE) meanRaw += input[i]
        meanRaw /= WINDOW_SIZE.toFloat()

        // 1) normalize (x-mean)/v10Sig
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

            val v = (r - meanRaw) / sig
            inputNorm[i] = v

            meanIn += v
            sumSq += v * v
            inMin = min(inMin, v)
            inMax = max(inMax, v)
        }
        meanIn /= WINDOW_SIZE.toFloat()
        val varIn = (sumSq / WINDOW_SIZE.toFloat()) - meanIn * meanIn
        val stdIn = sqrt(max(varIn, 0f))

        Log.d(TAG, "Input raw range=[$rawMin, $rawMax], raw mean=$meanRaw")
        Log.d(TAG, "Input norm mean=$meanIn, std=$stdIn, range=[$inMin, $inMax], v10Sig=$sig")

        // ✅ 關鍵：依 dimNames=[SequenceLength, BatchSize, Feature] → 固定餵 [2000, 1, 1]
        // 這樣 SequenceLength=2000，BatchSize=1，Feature=1，符合你 MATLAB 1×2000 的語意
        val inputShape = longArrayOf(WINDOW_SIZE.toLong(), 1L, 1L)
        Log.d(TAG, "Using input shape=${inputShape.contentToString()} (SeqLen, Batch, Feature)")

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

                // 大多輸出就是 2000；若多於 2000，取最後 2000
                val start = if (n == WINDOW_SIZE) 0 else (n - WINDOW_SIZE)

                val gfMin = currentModel.gfMin
                val gfMax = currentModel.gfMax
                val span = (gfMax - gfMin)

                for (i in 0 until WINDOW_SIZE) {
                    val y = tmp[start + i]   // flow_nm, ideally in [-1,1]
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

        Log.d(TAG, "Model y range=[$yMin, $yMax] -> Output SLM range=[$outMin, $outMax], mean=$outMean")
        Log.i(TAG, "Pred SLM min/max = [$outMin, $outMax]")

        return outputSlm
    }

    private fun md5(data: ByteArray): String {
        val md = MessageDigest.getInstance("MD5")
        return md.digest(data).joinToString("") { "%02x".format(it) }
    }
}
