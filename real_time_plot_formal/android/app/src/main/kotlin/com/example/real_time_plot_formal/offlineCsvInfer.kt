package com.example.real_time_plot_formal

import android.content.Context
import android.util.Log
import kotlin.math.max
import kotlin.math.min
import kotlin.math.PI
import kotlin.math.sin

class OfflineCsvInfer(
    private val context: Context,
    private val onnx: OnnxModelInfer
) {

    companion object {
        private const val TAG = "OFFLINE_INFER"
        private const val FS = 1000
        private const val WINSIZE = 2000
        private const val STRIDE = 200
        private const val COL_PIEZO = 1   // raw(:,2) in MATLAB (0-based)
    }

    // ============================================================
    // Public API
    // ============================================================
    fun runChiruStand03(): FloatArray {
        val rawPiezo = loadPiezoFromCsv("testdata/chirustand03.csv")
        Log.i(TAG, "Loaded CSV samples=${rawPiezo.size}")

        val v10 = butterworth10HzZeroPhase(rawPiezo)
        val N = v10.size

        val predSum = FloatArray(N)
        val wSum = FloatArray(N)

        val dropRatio = 0.05f
        val dropEdge = (dropRatio * WINSIZE).toInt()

        val taperF = FloatArray(dropEdge) { i -> i.toFloat() / dropEdge }
        val taperB = FloatArray(dropEdge) { i -> taperF[dropEdge - 1 - i] }

        var s = 0
        while (s + WINSIZE <= N) {
            val win = v10.copyOfRange(s, s + WINSIZE)

            // === 核心：直接呼叫你已驗證過的 infer() ===
            val y = onnx.infer(win)

            for (i in 0 until WINSIZE) {
                var w = 1f
                if (dropEdge > 0) {
                    if (i < dropEdge) w = taperF[i]
                    else if (i >= WINSIZE - dropEdge) w = taperB[i - (WINSIZE - dropEdge)]
                }
                predSum[s + i] += y[i] * w
                wSum[s + i] += w
            }
            s += STRIDE
        }

        val out = FloatArray(N)
        for (i in 0 until N) {
            out[i] = if (wSum[i] > 0f) predSum[i] / wSum[i] else 0f
        }

        Log.i(TAG, "Offline inference finished")
        return out
    }

    // ============================================================
    // CSV loader
    // ============================================================
    private fun loadPiezoFromCsv(assetName: String): FloatArray {
        val lines = context.assets.open(assetName)
            .bufferedReader()
            .use { it.readLines() }

        val data = ArrayList<Float>(lines.size)

        for (line in lines) {
            if (line.isBlank()) continue
            val cols = line.split(",")
            if (cols.size <= COL_PIEZO) continue

            val v = cols[COL_PIEZO].toFloatOrNull()
            if (v != null) data.add(v)
        }
        return data.toFloatArray()
    }

    // ============================================================
    // 10 Hz Butterworth (zero-phase, MATLAB filtfilt equivalent)
    // ============================================================
    private fun butterworth10HzZeroPhase(x: FloatArray): FloatArray {
        // 2nd-order Butterworth, same as MATLAB butter(2,10/(fs/2))
        val b = floatArrayOf(
            0.0009446918f,
            0.0018893836f,
            0.0009446918f
        )
        val a = floatArrayOf(
            1.0f,
            -1.9111971f,
            0.91497583f
        )

        val y = iirFilter(x, b, a)
        val yr = y.reversedArray()
        val yr2 = iirFilter(yr, b, a)
        return yr2.reversedArray()
    }

    private fun iirFilter(x: FloatArray, b: FloatArray, a: FloatArray): FloatArray {
        val y = FloatArray(x.size)
        for (n in x.indices) {
            y[n] = b[0] * x[n]
            if (n >= 1) {
                y[n] += b[1] * x[n - 1] - a[1] * y[n - 1]
            }
            if (n >= 2) {
                y[n] += b[2] * x[n - 2] - a[2] * y[n - 2]
            }
        }
        return y
    }
}
