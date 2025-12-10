// 更新 DataPoint 類以包含原始數據
class DataPoint {
  double t;
  double y;
  double originalY; // 添加原始數據

  DataPoint(this.t, this.y, this.originalY);
}
