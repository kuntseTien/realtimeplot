import onnx

m = onnx.load(
    r"C:\Users\kuntse\Desktop\NTHU\Kuntse\Flutter\real_time_plot_formal\STAND.onnx"
)

print("IR version:", m.ir_version)
print("Opset imports:")
for opset in m.opset_import:
    print(opset)
