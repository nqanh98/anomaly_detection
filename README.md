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
- img_mask_index.png
  - 抽出されたモジュールの画像
- img_mask_index_no_dbscan.png
  - 抽出されたモジュールの画像（DBSCANによるゴミ除去をかける前）
- module_contours.pkl
  - モジュールの輪郭情報
- module_labels.pkl
  - モジュールのラベル情報（グループサイズ調整前）
- module_labels_split.pkl
  - モジュールのラベル情報（グループサイズ調整後）

### 異常検知の実行
モジュール検出の実行後、以下のコマンドを実行してノートブックを起動し、上から順番にプログラムを実行してください</br>
```
jupyter notebook test_anomaly_detection.ipynb
```
全てのプログラムの実行後、以下のデータがoutputsフォルダに出力されます</br>
- img_target_index.png
  - 異常モジュールに色枠をつけた画像
- anomaly_modules.json
  - 異常タイプごとにモジュール番号を格納したもの

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
    - その他の細々としたプログラムを纏めたもの
- lib (外部ライブラリ)
  - contours_extractor.py
    - モジュール検出用のライブラリ（白石コードに対応するもの）
  - xmeans.py
    - X-meansの外部ライブラリ（テスト用に実装したもので、現状は使用しておりません）
  - star_clustering.py
    - Start clusteringの外部ライブラリ（テスト用に実装したもので、現状は使用しておりません）
- params (モジュール検出用のパラメータ)
  - upper_lim_pix_val.npy
  - lower_lim_pix_val.npy

## Parameters
異常検知 (anomaly_detection.py) における各種パラメータの説明</br>

### HotspotDetectors

#### LocalOutlierFactorモデルに関するパラメータ
- offset_lof
  - LocalOutlierFactorモデルの閾値(default: -4.0)
  - 小さくするほどホットスポットの検出力が上がるが誤検出も増える

#### IsolationForestモデルに関するパラメータ
- offset_isof
  - IsolationForestモデルの閾値(default: -0.75)
  - 小さくするほどホットスポットの検出力が上がるが誤検出も増える

#### RobustZscoreモデルに関するパラメータ
- gamma
  - モジュール単位のzスコアにおけるガンマ補正の係数(default: 3.0)
  - 大きくするほど高温領域を強調することが可能だが誤検出も増える
- min_zscore
  - モジュール単位のzスコアにおいて異常と判定するzスコアの閾値　
  - この値以上よりも大きいzスコアを持つクラスタをホットスポットと判定する(default: 4.0)

### AnomalyTypeClassifier

#### ホットスポットの形状分析に関するパラメータ
- min_hotspot_size
  - ホットスポットの最小サイズ
  - この値よりも小さいクラスタはホットスポットから除外する(default: 4)
- min_circularity
  - ホットスポットの最小真円度
  - この値よりも小さい真円度のクラスタはホットスポットから除外する(default: 0.25)
- min_waveness_shape_factor
  - ホットスポットの最小
  - これ値よりも小さいwavensss shape factorのクラスタはホットスポットから除外する(default: 0.7)
（備考）パラメータの詳細は[Shape factor](https://en.wikipedia.org/wiki/Shape_factor_(image_analysis_and_microscopy))を参考にすること

#### モジュール異常判定に関するパラメータ
- min_module_anomaly_ratio
  - モジュールにおけるホットスポットの割合がこの値よりも大きい時、モジュール異常と判定される(default: 0.5)

#### クラスタ異常判定に関するパラメータ
- min_cluster_anomaly_ratio
  - モジュールにおけるホットスポットの割合がこの値よりも大きく、かつ長軸方向へ縦長のホットスポットの時、クラスタ異常と判定される(default: 0.2)
- cluster_anomaly_offset
  - クラスタ異常とみなす長軸方向の長さのオフセットを割合で指定する(default: 0.2)
  - 大きくするほど短めのホットスポットでもクラスタ異常とみなすようになるが誤検出も増える

#### ジャンクションボックス異常判定に関するパラメータ
- junction_box_offset_long
  - 長軸方向のジャンクションボックス領域に関係する(default: 0.2)
  - 大きくするほど長軸方向のジャンクションボックス領域が広がる
- junction_box_offset_short
  - 短軸方向のジャンクションボックス領域を割合で指定する(default: 0.3)
  - 大きくするほど短軸方向のジャンクションボックス領域が広がる
- junction_box_offset_count
  - ジャンクションボックス領域からはみ出た点をどれくらい許容するかを指定する（default: 12)
