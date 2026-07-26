# ChArUco Generator / Detector

## ファイル

- `detect_charuco.py`
  - 現行のROS 2 ChArUco検出器。
  - 検出したArUcoマーカーIDから、TFの子フレームを自動で割り当てる。
  - 検出したChArUcoフレームから、固定のカメラリンクTFを配信する。
    - `left_camera_charuco` -> `left_camera_link`
    - `right_camera_charuco` -> `right_camera_link`
    - これらは `publish_world_camera_tf:=false` のときだけ配信される。
  - `world` が引ければ、world基準のカメラリンクTFも配信する。
    - `world` -> `left_camera_link`
    - `world` -> `right_camera_link`
    - RealSenseは各 `*_camera_link` からカメラ内部のフレームを配信する。
  - 検出したworld基準のカメラTFを保存し、起動時に読み直す。
    - 既定のキャッシュ: `charuco_ros2/config/world_camera_tfs.json`
    - 検出に失敗したときは、保存済みのTFを代わりに配信する。
    - オフセット: ChArUcoフレームから `x=0.052 m`, `y=0.067 m`, `z=0.0 m`
    - 回転: ChArUcoフレームから `z=90 deg` → `x=90 deg` → `y=180 deg`
- `multi_cube_charuco_detector.py`
  - 2個のキューブそれぞれの5面を検出する。詳細は後述。
  - 検出した多面ChArUcoから `left_camera_link` / `right_camera_link` を推定し、
    `world_camera_tfs.json` を更新する。
- `generate_charuco_5_7.py`
  - 現行の7x5 ChArUcoボード2枚のジェネレータ。
    - `left_camera_charuco`, マーカーID `0-16`
    - `right_camera_charuco`, マーカーID `30-46`
- `move_to_charuco.py`
  - 検出したキューブの手前へxArm6を移動させる。詳細は後述。
- `sweep_cube_targets.py`
  - アーム周囲の多数の目標でIKを解き、怪しい動きを洗い出す。詳細は後述。
- `detect_apriltag.py`
  - AprilTag検出スクリプト。
- `detect_charuco_9_7.py`
  - 旧9x7 ChArUco検出器。参考用に残している。
- `generate_charuco_9_7.py`
  - 旧9x7 ChArUcoジェネレータ。参考用に残している。

## 出力ディレクトリ

- `boards/generated/`
  - `detect_charuco.py` が使う現行の生成ボード。
- `boards/legacy/`
  - 旧世代のボード画像/PDF。参考用に残している。

## 現行ボードの生成

```bash
python3 generate_charuco_5_7.py
```

## ROS 2ノードの実行

```bash
colcon build --packages-select charuco_ros2
source install/setup.bash

ros2 run charuco_ros2 charuco_detector
```

検出器が配信するTF:

- 画像フレーム -> `left_camera_charuco` または `right_camera_charuco`
- `left_camera_charuco` -> `left_camera_link`（`publish_world_camera_tf:=false` のとき）
- `right_camera_charuco` -> `right_camera_link`（`publish_world_camera_tf:=false` のとき）
- `world` -> `left_camera_link`
- `world` -> `right_camera_link`

RealSenseを `publish_tf:=true` で動かすと、ドライバが以下のような内部フレームを配信する。

- `left_camera_link` -> `left_camera_depth_optical_frame`
- `left_camera_link` -> `left_camera_color_optical_frame`
- `right_camera_link` -> `right_camera_depth_optical_frame`
- `right_camera_link` -> `right_camera_color_optical_frame`

固定のカメラリンクのオフセットと回転は、ROS 2パラメータで変更できる。

```bash
ros2 run charuco_ros2 charuco_detector --ros-args \
  -p camera_link_offset_x:=0.052 \
  -p camera_link_offset_y:=0.067 \
  -p camera_link_offset_z:=0.0 \
  -p camera_link_rotation_z_deg:=90.0 \
  -p camera_link_rotation_x_deg:=90.0 \
  -p camera_link_rotation_y_deg:=180.0
```

world基準のTF配信も、ROS 2パラメータで変更できる。

```bash
ros2 run charuco_ros2 charuco_detector --ros-args \
  -p world_frame:=world \
  -p publish_world_camera_tf:=true \
  -p world_tf_cache_file:=charuco_ros2/config/world_camera_tfs.json \
  -p saved_world_tf_publish_rate:=10.0
```

起動時、検出器は `world_tf_cache_file` から保存済みTFを読み込む。
ChArUcoボードを検出すると、検出した `world` TFがキャッシュを上書きする。
検出に失敗したときは、失敗理由をログに出し、保存済みTFを配信する。

配信中のworldカメラTFの確認:

```bash
ros2 run tf2_ros tf2_echo world left_camera_link
ros2 run tf2_ros tf2_echo world left_camera_depth_optical_frame
ros2 run tf2_ros tf2_echo world right_camera_link
ros2 run tf2_ros tf2_echo world right_camera_depth_optical_frame
```

補助的な検出ノード:

```bash
ros2 run charuco_ros2 charuco_detector_9_7
ros2 run charuco_ros2 apriltag_detector
```

## マルチキューブ検出器（`multi_cube_charuco_detector`）

50 mmキューブ2個について、それぞれ5面のChArUcoを検出し、
`left_camera_link` / `right_camera_link` を更新する。

```bash
ros2 run charuco_ros2 multi_cube_charuco_detector
```

起動中は保存済み `world -> *_camera_link` TFを配信し続ける。画像検出は
CPU負荷を抑えるため既定では常時回さず、Triggerサービスを呼んだときだけ
`detection_window_sec` 秒間実行する。

```bash
ros2 service call /multi_cube_charuco_detector/detect_once std_srvs/srv/Trigger
```

### 配信するTF

- 面のTF `*_charuco`（例: `left_cube_front_charuco`）
  - カメラ光学フレーム基準の生の観測。検出できたフレームだけ配信する。
- front補助TF `left_cube_front` / `right_cube_front`
  - 検出した面の左角TFからfront面へ変換したデバッグ用TF。
  - front面からcamera_linkへの実測オフセットを合成する。cube_link と
    camera_link は平行なので、front -> camera_link は計算上は並進だけを使う。
- camera_link TF `left_camera_link` / `right_camera_link`
  - 起動時は `world_camera_tfs.json` の保存値を配信する。
  - ChArUco検出時は、検出面 -> front -> camera_link で求めた
    `world -> *_camera_link` を再配信し、`world_camera_tfs.json` に保存する。
  - 同じキューブの複数面が同時に見えたときは、カメラに近い1面だけを使う。
    同距離ならChArUcoコーナー解、次にコーナー数・マーカー数で選ぶ。

### world基準のカメラリンクTFも配信する

起動時に `world_camera_tfs.json`（`detect_charuco` が検出・保存したもの）を読み、
`world -> left_camera_link` / `world -> right_camera_link` を**静的TF**として
配信する。これが無いと `world` から `left_camera_color_optical_frame` へ辿る辺が
誰も配信しておらず、左右カメラの**点群をworld座標に変換できない**。

```
world ──> link_base ──> ...            robot_state_publisher（xarmのURDF）
world ──> left_camera_link ──> ...     このノード（またはdetect_charuco）
```

- 既定のキャッシュ: `charuco_ros2/config/world_camera_tfs.json`
- 起動時は保存値を読む。cubeを検出したら、対応するcamera_linkのTFを
  再計算して再配信し、既定では `world_camera_tfs.json` に保存する。
- `detect_charuco` と同時に起動する場合は、同じ辺を二重に配信することに
  なるので `publish_world_camera_tf:=false` にする。

### 検出の律速はカメラの解像度

ボードは4x4マスでマーカーは7 mm。ArUcoマーカーは枠込みで6セル分の幅がある。
IDの復号には**1セルあたり4〜5 px**が目安になる。1セルあたりのピクセル数は
`marker_length * fx / distance / 6` なので、ハンドカメラの実測値
（424x240で `fx = 310`）だとこうなる。

| カラープロファイル | fx | 0.15 m | 0.20 m | 0.30 m |
| --- | --- | --- | --- | --- |
| 424x240 | 310 | 2.4 px | **1.8 px** | 1.2 px |
| 640x360 | 468 | 3.6 px | 2.7 px | 1.8 px |
| 848x480 | 620 | 4.8 px | 3.6 px | 2.4 px |
| 1280x720 | 936 | 7.3 px | **5.5 px** | 3.6 px |
| 1920x1080 | 1404 | 10.9 px | 8.2 px | 5.5 px |

424x240では復号限界を下回っており、検出は安定しない。
`argus octomap_charuco_workflow.launch.py` から起動した場合は、最終姿勢で
ChArUco検出サービスを実行している間だけ、カラーストリームを
`1280x720x30`へ切り替える。検出後は成功・失敗にかかわらず、元の
プロファイルへ戻す。

RealSense ROSではプロファイル変更後にストリームの再有効化が必要なため、
検出器は `enable_color=false`、プロファイル変更、`enable_color=true` の
順で更新し、1280x720の新しい`camera_info`を受け取ってから検出を始める。

この動作はlaunch引数で変更できる。

```bash
ros2 launch argus octomap_charuco_workflow.launch.py \
  manage_detection_color_profile:=true \
  realsense_node_name:=/camera/hand_camera \
  detection_color_profile:=1280x720x30
```

ログに `CameraInfo updated: 1280x720` が出ず、`fx` が310のままなら、
RealSenseノード名または対応プロファイルを確認すること。

マーカーを大きくする方向では解決しない。424x240のまま0.20 mで復号するには
19.4 mmのマーカー、つまり110 mmのボードが要るが、50 mmのキューブ面には載らない。

### 姿勢推定のフォールバック（`allow_marker_only_pose`、既定 `true`）

4x4のChArUcoボードは、内部チェス盤コーナーが**9点しかない**。しかもコーナーを
1点補間するだけでも、その周囲のマーカーが解像されている必要がある。内部コーナー
4点を必須にすると、**マーカーは正しく読めているのにコーナーが揃わない**という
理由だけでフレームを捨てることになる。

姿勢推定は2段構えにしてある。

1. `chessboard` — 内部コーナーが4点以上あればそれを使う。精度が高いので優先する。
2. `marker` — マーカーの角そのものを使う。マーカー1個につき4点得られるので、
   `min_markers_for_pose`（既定2）個あれば足りる。

デバッグ画像のラベルにどちらで解いたかが出る。
例: `left_cube left z=0.210m [chessboard]`

どちらも `SOLVEPNP_IPPE`（平面専用のソルバ）を使う。ボード面上の点はすべて
同一平面であり、`SOLVEPNP_ITERATIVE` は平面かつ点数が少ないと不安定になるため。

| パラメータ | 既定値 | 意味 |
| --- | --- | --- |
| `allow_marker_only_pose` | `true` | マーカーの角へフォールバックする |
| `min_markers_for_pose` | `2` | フォールバックに必要なマーカー数 |

## ハンドカメラ仲介の外部カメラ検証

外部カメラと校正済みハンドカメラから同じ検証用ChArUcoボードを観測し、
それぞれの `base_link -> target` を比較する暫定チェック用ノード群。

検証用ボードを生成する。既定値はハンドカメラでも安定して読めるように
4x4 / square 20 mm / marker 14 mm / `DICT_4X4_100` で、既存キューブの
ID `0-79` と衝突しないように `80-87` を使う。

```bash
ros2 run charuco_ros2 generate_validation_charuco_board
```

生成物は既定で `charuco_ros2/boards/generated/validation_charuco/` に保存される。
PNG/PDFを印刷し、同時に出るYAMLまたはJSONを検出ノードの
`board_config_path` に渡す。`board_config_path` が `charuco_ros2/` で始まる
相対パスの場合、検出ノードは実行ディレクトリではなくパッケージ内パスとして解決する。

単体検出ノード:

```bash
ros2 run charuco_ros2 charuco_target_detector --ros-args \
  -p image_topic:=/left_camera/color/image_raw \
  -p camera_info_topic:=/left_camera/color/camera_info \
  -p board_config_path:=charuco_ros2/boards/generated/validation_charuco/validation_charuco_4x4_square20mm_marker14mm_id80-87_300dpi.yaml \
  -p output_frame_id:=target_pose_ext_left \
  -p pose_topic:=/charuco_validation/ext_pose
```

検出ノードは `camera_optical_frame -> output_frame_id` のTFと、
同じ姿勢の `geometry_msgs/PoseStamped` をpublishする。

外部カメラ検出、ハンドカメラ検出、比較ノードをまとめて起動する例:

```bash
ros2 launch charuco_ros2 charuco_target_validation.launch.py \
  board_config_path:=charuco_ros2/boards/generated/validation_charuco/validation_charuco_4x4_square20mm_marker14mm_id80-87_300dpi.yaml \
  ext_image_topic:=/left_camera/color/image_raw \
  ext_camera_info_topic:=/left_camera/color/camera_info \
  ext_output_frame_id:=target_pose_ext_left \
  hand_image_topic:=/camera/hand_camera/color/image_raw \
  hand_camera_info_topic:=/camera/hand_camera/color/camera_info \
  hand_output_frame_id:=target_pose_hand \
  base_frame:=base_link \
  validation_side:=left \
  csv_path:=/tmp/charuco_validation_left.csv
```

比較ノードは `/charuco_validation/ext_pose` と
`/charuco_validation/hand_pose` を `base_frame` へ変換し、並進誤差[mm]と
回転誤差[deg]をログ出力する。`csv_path` を指定すると表IIの集計に使える
CSVを追記する。検証ターゲットTFの古い時刻で外挿エラーになるのを避けるため、
launch既定ではPoseStamped topic同士を比較する。

比較を始める前に、`validation_side:=left` なら
`[48, 0, -77, 0, 77, 0]` deg、`validation_side:=right` なら
`[-48, 0, -77, 0, 77, 0]` degへMoveGroup経由で移動する。その後、
端末でEnterを押すと比較とCSV追記が始まる。

## キューブへの移動（`move_to_charuco`）

検出したキューブの手前の待機姿勢へ、xArm6を移動させる。

```bash
ros2 run charuco_ros2 move_to_charuco --ros-args -p execution_mode:=sim
ros2 service call /move_to_charuco/cube_left std_srvs/srv/Trigger "{}"
```

pre-marker姿勢 `[0, -15, 0, 0, -90, 0]` degへ戻す専用サービス:

```bash
ros2 service call /move_to_charuco/pre_marker std_srvs/srv/Trigger "{}"
```

`execution_mode:=sim` は、MoveItの `/compute_ik` サービスでポーズを解き、
得られた関節角を MoveGroup (`xarm_utils_py`) の `set_joint_value_target()` →
`plan()` → `execute()` で実行する。

### simとrealを一致させる（`real_use_moveit_ik`、既定 `true`）

`execution_mode:=real` も同じ `/compute_ik` を呼び、得られた関節角をMoveGroupの
`plan()` → `execute()` で実行する。IKの分岐も実行経路も、simで見たものと一致する。

`real_use_moveit_ik` は後方互換のため残しているが、現在は `false` にしても
xArmの `set_position` サービスへはフォールバックしない。planningまたはexecuteが
失敗した場合、サービス応答は非0の `ret` を返す。

`cartesian_max_target_distance_m` は両モードで同じ値を渡すこと。この値は目標位置
そのものを変えるので、違う値では同じポーズを比較していることにならない。

### 手先を水平に保つ（`cartesian_level_tool_forward`、既定 `true`）

待機位置はキューブより `cube_target_z_offset_m` だけ下にあるので、手先をキューブへ
まっすぐ向けると上向きに傾く。スタンドオフ5 cm・高さオフセット-5 cmだと、手先は
**45度の上向き**を要求される。するとIKは、joint1がキューブと逆を向き、肩越しに
後ろへ反り返る分岐を返してしまう。

`cartesian_level_tool_forward` は、向きベクトルから上下成分を落とし、高さオフセット
は保ったまま手先を水平にする。有効にすると、IKは素直な分岐を返す。方位98度の
キューブの場合:

```
IK solution: solution_deg=[98.0, 41.6, -114.0, -0.0, -17.6, 0.0]
```

joint1がキューブの方位と一致し、joint4とjoint6はほぼ0度。手首が捻れない。

### IKのシード（`sim_ik_seed_mode`、既定 `current_or_side`）

KDLはシードに最も近い解へ収束する反復ソルバなので、同じポーズを実現する
（最大8通りの）関節解のうちどれが返るかは、シードが決める。

- `current` — 常に現在の関節角をシードにする。
- `current_or_side` — 現在の関節角。取れなければ決め打ちのside姿勢（既定）。
- `side_pose` — 常に決め打ちのside姿勢。ロボットが実際にどこにいるかと無関係
  なので、現在姿勢から遠い分岐を選びやすい。非推奨。

### joint1の事前回転（`align_joint1_before_move`、既定 `false`）

残りの関節を解く前に、joint1だけを目標方位へ回す。ただしこれは総移動量を
**減らさない**。IKがjoint1を回し戻す分岐を返すことがあり、その場合、事前回転は
動きを増やすだけになる。実験用に残してあるが、正面向きの分岐が存在すると
分かっている場合以外はオフのままにすること。

## 目標のスイープ（`sweep_cube_targets`）

アームの周囲の多数の位置に仮想のキューブを置き、それぞれのIK解を報告する。
`move_to_charuco` と同じ接近位置・姿勢の計算を通す。既定ではIKを解くだけで
アームは動かさない。

```bash
ros2 run charuco_ros2 sweep_cube_targets --ros-args \
  -p execution_mode:=sim \
  -p cartesian_max_target_distance_m:=0.5 \
  -p server_service_name:=/move_to_charuco_sweep
```

`server_service_name` を変えているのは、`move_to_charuco` が同時に動いていても
サービス名が衝突しないようにするため。

各目標には次のタグが付く。

- `BACKWARD` — joint1が目標方位から90度以上ずれている。肩越しに後ろへ反り返る分岐。
- `FAR` — いずれかの関節が150度以上動く。
- `WRIST` — joint4かjoint6が120度以上捻れる。
- `NO_IK` — 解なし。

移動量は関節角の**生の差**であり、±180度に折り返した値ではない。コントローラは
関節値を線形補間するだけで、近いほうへ回り込んだりしないため。

RViz上で動きを目視したい場合は、1点ごとに固定の初期姿勢へ戻す。

```bash
ros2 run charuco_ros2 sweep_cube_targets --ros-args \
  -p execution_mode:=sim \
  -p cartesian_max_target_distance_m:=0.5 \
  -p server_service_name:=/move_to_charuco_sweep \
  -p sweep_execute:=true \
  -p sweep_dwell_time:=1.5 \
  -p sweep_home_joint_degrees:="[0.0, -15.0, 0.0, 0.0, -90.0, 0.0]"
```

`sweep_home_joint_degrees` は必ず渡すこと。渡さないと、起動時にアームがたまたま
いた姿勢が初期姿勢になり、再現性がない。初期姿勢はIKのシードでもあるので、
移動量に強く効く。

グリッドのパラメータ:

| パラメータ | 既定値 | 意味 |
| --- | --- | --- |
| `sweep_azimuth_step_deg` | `45.0` | ベース周りの方位の刻み |
| `sweep_cube_radii_m` | `[0.35, 0.45, 0.55]` | キューブの水平距離 |
| `sweep_cube_heights_m` | `[0.2, 0.4, 0.6]` | キューブの高さ |
| `sweep_execute` | `false` | 実際にアームを動かす |
| `sweep_return_home` | `true` | 各目標の後に初期姿勢へ戻る |
| `sweep_dwell_time` | `1.0` | 各姿勢での静止時間（秒） |
| `sweep_home_joint_degrees` | `[]` | 初期姿勢兼IKシード。空なら起動時の姿勢 |

### 既知の死角

xArm6のURDFで実測（joint1は±178.2度、joint5は-97.0〜178.2度）。

- **真後ろ（方位180度付近）は解けない。** 真後ろを向くにはjoint1が180度必要だが、
  上限を超えている。
- **低い目標は解けない。** 高さ0.20 mのキューブは手先を0.15 mに置くことになり、
  手先を水平に保つ限り、どの方位でも解が存在しない。
- **背後寄りの目標はjoint1が大きく振れる。** joint1は±178.2度を越えて回り込めない
  ので、短いほうへ回る等価角が存在しない。
