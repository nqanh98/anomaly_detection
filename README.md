# rule_based_anomaly_detection

## How to use

### Inputs
images/orthoフォルダに分析用のオルソ画像を格納してください</br>

### Outputs
outputsファルダに分析結果が出力されます</br>

### モジュール検出の実行
以下のコマンドを実行してノートブックを起動し、上から順番にプログラムを実行してください</br>
```
jupyter notebook test_module_extraction.ipynb
```
全てのプログラムの実行後、以下のデータがoutputsフォルダに出力されます</br>

### 異常検知の実行
モジュール検出の実行後、以下のコマンドを実行してノートブックを起動し、上から順番にプログラムを実行してください</br>
```
jupyter notebook test_anomaly_detection.ipynb
```
全てのプログラムの実行後、以下のデータがoutputsフォルダに出力されます</br>
異常と判定されたモジュールは、以下のルールに基づいて色づけされています</br>
```
color_list = {
    "Single-Hotspot": (0,255,255), # aqua
    "Multi-Hotspots": (0,255,0), # green
    "Cluster-Anomaly": (255,255,0), # yellow
    "Module-Anomaly": (255,165,0), # orange
    "String-Anomaly": (238,130,238) # violet
}
```

## Source codes
- images (分析対象の画像データ)
  - ortho/
    - 分析用のオルソ画像をここに置いてください
- notebooks (実行用のノートブック)
  - test_module_extraction.ipynb
    - モジュール検出用のノートブック
  - test_anomaly_detection.ipynb
    - 異常検知用のノートブック
- src (異常検知用のソースコード)
  - dataloader.py
    - データローダー
  - module_extraction.py
    - モジュール検出用のプログラム
  - anomaly_detection.py
    - 異常検知用のプログラム
  - clustering.py
    - クラスタリング用のプログラム
  - utils.py
    - その他細々としたプログラムを纏めたもの
- lib (外部ライブラリ)
  - contours_extractor.py
    - モジュール検出用のライブラリ（白石コードに対応するもの）
  - xmeans.py
    - X-meansの外部ライブラリ（テスト用に実装したもので、実際には使用しておりません）
  - star_clustering.py
    - Start clusteringの外部ライブラリ（テスト用に実装したもので、実際には使用しておりません）
- params (モジュール検出用のパラメータ)
  - upper_lim_pix_val.npy
  - lower_lim_pix_val.npy
- outputs (結果の出力先)
- old（不要コードの置き場）

## Parameters
異常検知 (anomaly_detection.py) における各種パラメータの説明</br>

### HotspotDetectors
ホットスポットの検出</br>
- gamma
  - (default: 1.5)
- alpha_lof
  - (default: -1.6)
- beta_lof
  - (default: 0.5)
- alpha_isof
  - (default: -0.6)
- beta_isof
  - (default: 0.2)
 
### AnomalyTypeClassifier
ホットスポットの形状分析</br>
- min_hotspot_size
  - (default: 4)
- min_circularity
  - (default: 0.25)
- min_waveness_shape_factor
  - (default: 0.7)

異常タイプの分類</br>
- min_module_anomaly_size
  - (default: 0.5)
- min_cluster_anomaly_size
  - (default: 0.25)
- gamma
  - (default: 3.0)
- min_zscore
  - (default: 3.0)
- junction_box_offset_long
  - (default: 0.2)
- junction_box_offset_short
  - (default: 0.3)
- junction_box_offset_count
  - (default: 12)
- cluster_anomaly_offset
  - (default: 0.1)