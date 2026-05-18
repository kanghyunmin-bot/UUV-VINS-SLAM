#!/usr/bin/env python3
from __future__ import annotations

import json
import csv
import os
import shlex
import socket
import shutil
import subprocess
import sys
import time
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
RECOMMENDED_FEATURE_METHOD = "shi_tomasi_klt"
FEATURE_METHOD_ORDER = {
    "shi_tomasi_klt": 0,
    "sift_match": 1,
    "akaze_match": 2,
    "orb_match": 3,
}


INDEX_HTML = r"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Underwater VINS-Fusion Lab</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f5f7f8;
      --panel: #ffffff;
      --text: #172026;
      --muted: #5c6970;
      --line: #d7dee2;
      --accent: #0f766e;
      --accent-dark: #115e59;
      --danger: #b42318;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
    }
    header {
      padding: 16px 22px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
      display: flex;
      align-items: baseline;
      gap: 14px;
    }
    h1 {
      margin: 0;
      font-size: 20px;
      letter-spacing: 0;
    }
    header span { color: var(--muted); font-size: 13px; }
    main {
      display: grid;
      grid-template-columns: 460px 1fr;
      gap: 16px;
      padding: 16px;
      min-height: calc(100vh - 58px);
    }
    aside, section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }
    aside {
      padding: 14px;
      max-height: calc(100vh - 90px);
      overflow: auto;
    }
    section {
      padding: 14px;
      min-width: 0;
    }
    label {
      display: block;
      margin: 10px 0 4px;
      font-size: 12px;
      color: var(--muted);
      font-weight: 600;
    }
    input, select {
      width: 100%;
      padding: 8px 9px;
      border: 1px solid var(--line);
      border-radius: 6px;
      font-size: 14px;
      background: #fff;
      color: var(--text);
    }
    .grid2 {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }
    .checks {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px 12px;
      margin: 12px 0;
    }
    .checks label {
      display: flex;
      align-items: center;
      gap: 8px;
      margin: 0;
      font-size: 13px;
      color: var(--text);
      font-weight: 500;
    }
    input[type="checkbox"] { width: auto; }
    .buttons {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      margin-top: 12px;
    }
    button {
      border: 0;
      border-radius: 6px;
      background: var(--accent);
      color: #fff;
      padding: 10px 12px;
      font-weight: 700;
      cursor: pointer;
    }
    button.secondary { background: #334155; }
    button.compact {
      padding: 8px 10px;
      font-size: 12px;
    }
    button:disabled {
      cursor: progress;
      opacity: 0.65;
    }
    hr {
      border: 0;
      border-top: 1px solid var(--line);
      margin: 16px 0;
    }
    .status {
      padding: 10px 12px;
      border-radius: 6px;
      background: #ecfdf5;
      color: var(--accent-dark);
      font-weight: 700;
      margin-bottom: 12px;
    }
    .status.error {
      background: #fef3f2;
      color: var(--danger);
    }
    .preview {
      width: 100%;
      min-height: 320px;
      background: #101820;
      border-radius: 8px;
      display: grid;
      place-items: center;
      overflow: hidden;
      border: 1px solid #111827;
    }
    .preview img {
      width: 100%;
      height: auto;
      display: block;
    }
    .preview video {
      width: 100%;
      max-height: 72vh;
      display: block;
      background: #000;
    }
    .preview-grid {
      display: grid;
      grid-template-columns: minmax(280px, 0.8fr) minmax(320px, 1.2fr);
      gap: 12px;
      align-items: start;
    }
    .preview-grid .preview {
      min-height: 260px;
    }
    .panel-title {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      margin: 0 0 8px;
      font-size: 13px;
      font-weight: 800;
      color: #25313a;
    }
    .help {
      margin-top: 6px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
    }
    .insight {
      margin-top: 10px;
      padding: 11px;
      border: 1px solid #b8d8d2;
      border-radius: 8px;
      background: #f0fdfa;
      font-size: 13px;
      line-height: 1.45;
    }
    .insight strong {
      display: block;
      margin-bottom: 5px;
      color: var(--accent-dark);
      font-size: 14px;
    }
    .badge {
      display: inline-flex;
      align-items: center;
      gap: 5px;
      padding: 2px 7px;
      border-radius: 999px;
      background: #ccfbf1;
      color: #134e4a;
      font-size: 11px;
      font-weight: 800;
    }
    .benchmark {
      margin-top: 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: #fbfcfc;
    }
    .benchmark h2 {
      margin: 0 0 6px;
      font-size: 15px;
      letter-spacing: 0;
    }
    .benchmark .lead {
      margin: 0;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
    }
    .benchmark tr.recommended {
      background: #ecfdf5;
    }
    .benchmark td.method {
      text-align: left;
      font-weight: 800;
    }
    .benchmark td.source {
      text-align: left;
      color: var(--muted);
      max-width: 260px;
      overflow-wrap: anywhere;
    }
    .mini-row {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 8px;
      align-items: end;
    }
    .summary {
      margin-top: 12px;
      display: grid;
      grid-template-columns: repeat(4, minmax(120px, 1fr));
      gap: 10px;
    }
    .metric {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      background: #fbfcfc;
    }
    .metric strong {
      display: block;
      font-size: 18px;
      margin-top: 4px;
    }
    .links, .notes {
      margin-top: 12px;
      font-size: 13px;
      line-height: 1.6;
    }
    pre.log {
      margin: 12px 0 0;
      padding: 12px;
      max-height: 320px;
      overflow: auto;
      border-radius: 8px;
      background: #0f172a;
      color: #dbeafe;
      font-size: 12px;
      line-height: 1.45;
      white-space: pre-wrap;
    }
    a { color: var(--accent-dark); }
    table {
      width: 100%;
      border-collapse: collapse;
      margin-top: 12px;
      font-size: 13px;
    }
    th, td {
      border-bottom: 1px solid var(--line);
      padding: 7px 6px;
      text-align: right;
    }
    th:first-child, td:first-child { text-align: left; }
    @media (max-width: 980px) {
      main { grid-template-columns: 1fr; }
      aside { max-height: none; }
      .summary { grid-template-columns: 1fr 1fr; }
      .preview-grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Underwater VINS-Fusion Lab</h1>
    <span>Stereo MP4 and ROSBag stereo+IMU VIO to RViz local path</span>
  </header>
  <main>
    <aside>
      <label>VINS Preset</label>
      <div class="grid2">
        <select id="vinsPresetSelect" onchange="applyVinsPreset()">
          <option value="vins_stereo_only_presentation_3100" selected>Presentation stereo-only, 3100, ~100m</option>
          <option value="vins_stereo_imu_precision_3100">Precision stereo+IMU, max 3100</option>
          <option value="vins_klt_reference">Stable tracking: Shi-Tomasi + KLT reference</option>
          <option value="vins_smoke">Smoke test, 100 frames</option>
          <option value="vins_pool_conservative">Pool conservative</option>
          <option value="vins_pool_sensitive">Pool sensitive</option>
          <option value="vins_full">Full stereo MP4</option>
        </select>
        <button class="secondary" type="button" onclick="applyVinsPreset()">Apply</button>
      </div>

      <label>Left MP4</label>
      <div class="mini-row">
        <select id="inputPath" onchange="updateSourcePreview()"></select>
        <button class="secondary compact" type="button" onclick="updateSourcePreview()">Reload</button>
      </div>

      <label>Right MP4</label>
      <select id="rightInputPath"></select>
      <div class="help">VINS-Fusion은 stereo pair 기준입니다. 여기서는 feature method를 고르지 않고 VINS의 tracker/config 파라미터만 조정합니다.</div>

      <div class="grid2">
        <div><label>start_frame</label><input id="startFrame" value="0"></div>
        <div><label>max_frames</label><input id="vinsMaxFrames" value="100"></div>
      </div>
      <div class="grid2">
        <div><label>stride</label><input id="vinsStride" value="1"></div>
        <div><label>resize_width</label><input id="vinsResizeWidth" value="848"></div>
      </div>
      <div class="grid2">
        <div><label>fps_override</label><input id="vinsFps" value="0"></div>
        <div><label>baseline_m</label><input id="vinsBaseline" value="0.05"></div>
      </div>

      <hr>
      <div class="panel-title">Camera Calibration</div>
      <div class="grid2">
        <div><label>focal_scale</label><input id="vinsFocalScale" value="0.62"></div>
        <div><label>fx explicit</label><input id="vinsFx" value="0"></div>
      </div>
      <div class="grid2">
        <div><label>fy explicit</label><input id="vinsFy" value="0"></div>
        <div><label>cx explicit</label><input id="vinsCx" value=""></div>
      </div>
      <div class="grid2">
        <div><label>cy explicit</label><input id="vinsCy" value=""></div>
        <div></div>
      </div>
      <div class="help">정확한 underwater stereo calibration이 없으면 trajectory는 참고용입니다. fx/fy/cx/cy를 모르면 focal_scale로 임시 생성합니다.</div>

      <hr>
      <div class="panel-title">VINS Feature Tracker</div>
      <div class="insight" id="trackerInsight"></div>
      <div class="grid2">
        <div><label>max_cnt</label><input id="vinsMaxCnt" value="150"></div>
        <div><label>min_dist</label><input id="vinsMinDist" value="30"></div>
      </div>
      <div class="grid2">
        <div><label>freq</label><input id="vinsFreq" value="10"></div>
        <div><label>F_threshold</label><input id="vinsFThreshold" value="1.0"></div>
      </div>

      <hr>
      <div class="panel-title">VINS Optimizer</div>
      <div class="grid2">
        <div><label>keyframe_parallax</label><input id="vinsKeyframeParallax" value="10.0"></div>
        <div><label>solver_time</label><input id="vinsMaxSolverTime" value="0.04"></div>
      </div>
      <div class="grid2">
        <div><label>solver_iterations</label><input id="vinsMaxIterations" value="8"></div>
        <div></div>
      </div>

      <hr>
      <div class="panel-title">ROSBag Stereo+IMU VIO</div>
      <label>rosbag2 db3</label>
      <input id="rosbagPath" value="/Users/kanghyunmin/Desktop/uuv_sim/real_robot_ros_bag/extracted_2026_04_01/bag_2026-04-01_20-20-30/bag_2026-04-01_20-20-30_0.db3">
      <div class="grid2">
        <div><label>imu_max_frames</label><input id="imuMaxFrames" value="150"></div>
        <div><label>sync_tolerance_ms</label><input id="imuSyncTolerance" value="3.0"></div>
      </div>
      <div class="grid2">
        <div><label>imu_source</label>
          <select id="imuSource">
            <option value="realsense" selected>RealSense gyro+accel</option>
            <option value="mavros">MAVROS /mavros/imu/data</option>
          </select>
        </div>
        <div><label>replay_realtime_factor</label><input id="replayRealtimeFactor" value="12"></div>
      </div>
      <div class="grid2">
        <div><label>estimate_extrinsic</label>
          <select id="estimateExtrinsic">
            <option value="1" selected>1 refine initial guess</option>
            <option value="0">0 fixed</option>
            <option value="2">2 calibrate no prior</option>
          </select>
        </div>
        <div><label>estimate_td</label>
          <select id="estimateTd">
            <option value="1" selected>1 estimate time offset</option>
            <option value="0">0 fixed td</option>
          </select>
        </div>
      </div>
      <div class="grid2">
        <div><label>acc_n</label><input id="accN" value="0.1"></div>
        <div><label>gyr_n</label><input id="gyrN" value="0.01"></div>
      </div>
      <div class="grid2">
        <div><label>acc_w</label><input id="accW" value="0.001"></div>
        <div><label>gyr_w</label><input id="gyrW" value="0.0001"></div>
      </div>
      <div class="panel-title">ROS2 Native Feature Method</div>
      <div class="grid2">
        <div><label>feature_detector</label>
          <select id="ros2FeatureDetector">
            <option value="orb" selected>ORB + KLT best</option>
            <option value="gftt">Shi-Tomasi/GFTT + KLT</option>
            <option value="harris">Harris + KLT</option>
            <option value="fast">FAST + KLT</option>
          </select>
        </div>
        <div><label>depth_scale</label><input id="ros2DepthScale" value="0.85"></div>
      </div>
      <div class="grid2">
        <div><label>max_features</label><input id="ros2MaxFeatures" value="480"></div>
        <div><label>min_features</label><input id="ros2MinFeatures" value="120"></div>
      </div>
      <div class="grid2">
        <div><label>cell_max_features</label><input id="ros2CellMaxFeatures" value="28"></div>
        <div><label>min_distance</label><input id="ros2MinDistance" value="16"></div>
      </div>
      <div class="grid2">
        <div><label>orb_fast_threshold</label><input id="ros2OrbFastThreshold" value="8"></div>
        <div><label>pnp_threshold</label><input id="ros2PnpThreshold" value="2.5"></div>
      </div>
      <div class="grid2">
        <div><label>temporal_ransac</label>
          <select id="ros2TemporalRansac">
            <option value="fundamental" selected>Fundamental</option>
            <option value="essential">Essential</option>
            <option value="none">None</option>
          </select>
        </div>
        <div><label>temporal_threshold</label><input id="ros2TemporalThreshold" value="0.6"></div>
      </div>
      <div class="grid2">
        <div><label>max_step_m</label><input id="ros2MaxStepM" value="0.22"></div>
        <div><label>translation_smoothing</label><input id="ros2TranslationSmoothing" value="0.12"></div>
      </div>
      <div class="panel-title">KLT Internal Logic</div>
      <div class="grid2">
        <div><label>klt_max_flow_px</label><input id="ros2KltMaxFlowPx" value="0"></div>
        <div><label>klt_patch_ncc</label><input id="ros2KltPatchNcc" value="0"></div>
      </div>
      <div class="grid2">
        <div><label>klt_patch_size</label><input id="ros2KltPatchSize" value="15"></div>
        <div><label>klt_max_accel_px</label><input id="ros2KltMaxAccelPx" value="25"></div>
      </div>
      <div class="grid2">
        <div><label>min_track_age_for_pnp</label><input id="ros2MinTrackAge" value="1"></div>
        <div><label>klt_min_eigenvalue</label><input id="ros2KltMinEigenvalue" value="0"></div>
      </div>
      <div class="checks">
        <label><input id="ros2RejectRepeatedPatches" type="checkbox"> reject repeated patches</label>
        <label><input id="ros2KltLocalFlowCheck" type="checkbox"> local flow check</label>
        <label><input id="ros2KltUsePrediction" type="checkbox"> prediction LK</label>
        <label><input id="ros2KltVelocityCheck" type="checkbox"> velocity jump check</label>
      </div>
      <div class="help">현재 best는 KLT 내부 추가 필터를 끈 ORB + KLT + Fundamental RANSAC입니다. KLT 필터는 분석용으로 켜볼 수 있지만, 이 bag에서는 너무 엄격하게 걸면 pose 성공률이 떨어졌습니다.</div>

      <hr>
      <div class="panel-title">RViz Live Playback</div>
      <div class="grid2">
        <div><label>odom_publish_hz</label><input id="rvizPublishHz" value="5.27"></div>
        <div><label>live CSV</label><input id="rvizLiveCsv" value="outputs/live/latest_odometry.csv"></div>
      </div>
      <div class="help">현재 RViz 비교 기준은 0~30초입니다. 파란 /local_path는 stereo+IMU VIO CSV만, 빨간 /sim_odom_reference_path는 DVL reference만 publish합니다.</div>

      <div class="panel-title">AQUA-style Metric Correction</div>
      <label>VIO input CSV</label>
      <input id="aquaInputCsv" value="outputs/vins_fusion/gui_runs/run_20260511_185018_097/vins_stereo_imu_odometry.csv">
      <div class="grid2">
        <div><label>metric_source</label>
          <select id="aquaMetricSource">
            <option value="reference" selected>/odometry/filtered reference</option>
            <option value="dvl">DVL twist integration</option>
            <option value="target">Manual target length</option>
          </select>
        </div>
        <div><label>target_path_m</label><input id="aquaTargetPathM" value="100"></div>
      </div>
      <div class="grid2">
        <div><label>jump_clamp_m</label><input id="aquaMaxStepM" value="0.3"></div>
        <div><label>output CSV</label><input id="aquaOutputCsv" value="outputs/live/latest_odometry_aqua_corrected.csv"></div>
      </div>
      <div class="help">VINS visual trajectory의 방향/형상은 유지하고, DVL 또는 EKF reference 거리로 metric scale과 큰 jump를 보정합니다. 완전한 tight-coupled AQUA-SLAM은 아니고 RViz/발표용 정확도 보정 단계입니다.</div>

      <div class="panel-title">DVL Odom Fit</div>
      <div class="grid2">
        <div><label>DVL start_sec</label><input id="dvlStartSec" value="0"></div>
        <div><label>DVL duration_sec</label><input id="dvlDurationSec" value="30"></div>
      </div>
      <div class="grid2">
        <div><label>target mean_corr</label><input id="dvlMinMeanCorr" value="0.70"></div>
        <div><label>fit modes</label><input id="dvlFitModes" value="rigid,sim3,affine"></div>
      </div>
      <div class="grid2">
        <div><label>max RMSE / path</label><input id="dvlMaxRmseRatio" value="0.08"></div>
        <div><label>max error / path</label><input id="dvlMaxErrorRatio" value="0.60"></div>
      </div>
      <div class="grid2">
        <div><label>DVL clean grid</label><input id="dvlCleanMaxStepGrid" value="0"></div>
        <div><label>VINS max_step grid</label><input id="dvlFitGrid" value="0.05,0.08,0.10,0.15,0.20,0.30,0.40,0.50,0.75,1.0,1.5,2.0,3.0,5.0,1000.0"></div>
      </div>
      <div class="help">현재 목표는 MP4 시작 기준 0~30초 DVL XYZ입니다. 빨간 경로는 DVL 기준선, 파란 경로는 선택한 fit/oracle 결과입니다. 0 clean은 DVL 값을 그대로 쓰고, oracle은 DVL 자체를 publish하는 상한선 확인용입니다.</div>

      <div class="panel-title">MuJoCo RC Replay Reference</div>
      <label>sim_bag db3</label>
      <input id="simRcBagPath" value="latest">
      <div class="grid2">
        <div><label>reference topic</label><input id="simRcReferenceTopic" value="/mavros/local_position/odom"></div>
        <div><label>time map</label>
          <select id="simRcTimeMap">
            <option value="normalized" selected>normalized full MP4</option>
            <option value="direct">direct seconds</option>
          </select>
        </div>
        <div><label>offset target</label>
          <select id="simRcOffsetTarget">
            <option value="source" selected>shift VINS, keep red ref fixed</option>
            <option value="reference">shift reference</option>
          </select>
        </div>
      </div>
      <div class="grid2">
        <div><label>fit modes</label><input id="simRcFitModes" value="rigid,sim3,affine"></div>
        <div><label>VINS max_step grid</label><input id="simRcFitGrid" value="0.03,0.05,0.08,0.10,0.12,0.15,0.20,0.30,0.40,0.50,0.75,1.0,1.5,2.0,1000.0"></div>
      </div>
      <label>time offset grid sec</label>
      <input id="simRcTimeOffsetGrid" value="-30,-24,-20,-16,-14,-12,-10,-8,-6,-4,-2,0,2,4,6,8,10,12,14,16,20,24,30">
      <div class="grid2">
        <div><label>metric axes</label>
          <select id="simRcMetricAxes">
            <option value="xy" selected>XY only</option>
            <option value="xyz">XYZ</option>
          </select>
        </div>
        <div><label>selection mode</label>
          <select id="simRcSelectionMode">
            <option value="max-corr" selected>maximize corr</option>
            <option value="quality">quality gates</option>
          </select>
        </div>
      </div>
      <div class="help">real rosbag의 MP4 구간 /mavros/rc/override를 MuJoCo에 replay해서 생긴 sim_bag만 기준으로 씁니다. 기본은 sim odom의 XY 경향성만 보고 z축은 평가/선택에서 제외합니다.</div>

      <div class="buttons">
        <button class="secondary" id="vinsBuildBtn" onclick="buildVinsDocker()">Build Docker</button>
        <button class="secondary" id="vinsConfigBtn" onclick="makeVinsConfig()">Make VINS Config</button>
        <button id="vinsRunBtn" onclick="runVins()">Run VINS</button>
        <button id="ros2VioRunBtn" onclick="runRos2StereoImuVio()">Run ROS2 Stereo+IMU VIO</button>
        <button class="secondary" id="vinsImuRunBtn" onclick="runStereoImuVins()">Run VINS-Fusion Stereo+IMU</button>
        <button id="aquaCorrectBtn" onclick="applyAquaCorrection()">Apply AQUA Correction</button>
        <button id="dvlFitBtn" onclick="fitToDvlOdom()">Evaluate VIO vs DVL</button>
        <button class="secondary" id="dvlOracleBtn" onclick="showDvlOracle()">Publish 30s DVL Ref Red</button>
        <button class="secondary" id="simRcReferenceBtn" onclick="publishFixedSimRcReference()">Publish Raw Sim Ref Red</button>
        <button id="simRcFitBtn" onclick="fitToSimRcReplay()">Fit to MuJoCo RC Replay</button>
        <button class="secondary" id="simRcOracleBtn" onclick="showSimRcOracle()">Show 100% Sim Ref Blue</button>
        <button class="secondary" id="rvizPublisherBtn" onclick="restartRvizPublisher()">Restart RViz Publisher</button>
      </div>
      <div class="help">Docker Desktop이 켜져 있어야 합니다. DVL은 estimator 입력으로 쓰지 않고 빨간 reference/사후 비교용으로만 둡니다. 파란 VIO 경로는 stereo+IMU 결과만 표시합니다.</div>
    </aside>

    <section>
      <div id="status" class="status">Ready</div>
      <div class="preview-grid">
        <div>
          <div class="panel-title">Input MP4 <span id="sourceInfo"></span></div>
          <div class="preview" id="sourcePreview"><span style="color:#cbd5e1">MP4를 선택하면 원본 영상이 표시됩니다.</span></div>
        </div>
        <div>
          <div class="panel-title">Run Result</div>
          <div class="preview" id="preview"><span style="color:#cbd5e1">Run VINS를 누르면 RViz용 odometry CSV가 생성됩니다.</span></div>
        </div>
      </div>
      <div class="summary" id="summary"></div>
      <div class="links" id="links"></div>
      <div class="benchmark" id="featureBenchmark">
        <h2>Feature Tracking Benchmark</h2>
        <p class="lead">저장된 실험 summary를 읽어 feature 방식별 best 결과를 표시합니다.</p>
      </div>
      <div class="benchmark" id="accuracyBenchmark">
        <h2>Trajectory Accuracy Benchmark</h2>
        <p class="lead">uuv_sim rosbag의 /odometry/filtered 기준 경로와 비교한 ATE 결과를 표시합니다.</p>
      </div>
      <div id="sweepResults"></div>
      <div class="notes">
        MP4 Run VINS는 stereo-only 비교용입니다. 실제 VIO 실험은 ROSBag Stereo+IMU VIO 버튼을 사용합니다.
      </div>
    </section>
  </main>
<script>
async function init() {
  const res = await fetch('/api/defaults');
  const data = await res.json();
  const leftSelect = document.getElementById('inputPath');
  const rightSelect = document.getElementById('rightInputPath');
  leftSelect.innerHTML = '';
  rightSelect.innerHTML = '';
  data.videos.forEach((video) => {
    const option = document.createElement('option');
    option.value = video.path;
    option.textContent = video.name;
    leftSelect.appendChild(option);
    rightSelect.appendChild(option.cloneNode(true));
  });
  const infra1 = data.videos.findIndex((video) => video.name.includes('infra1'));
  const infra2 = data.videos.findIndex((video) => video.name.includes('infra2'));
  if (infra1 >= 0) leftSelect.selectedIndex = infra1;
  if (infra2 >= 0) rightSelect.selectedIndex = infra2;
  applyVinsPreset();
  renderFeatureBenchmark(data.feature_benchmark);
  renderAccuracyBenchmark(data.accuracy_benchmark);
  attachTrackerListeners();
  updateSourcePreview();
}

const VINS_PRESETS = {
  vins_stereo_only_presentation_3100: {
    startFrame: 0, vinsMaxFrames: 3100, vinsStride: 1, vinsResizeWidth: 848, vinsFps: 0,
    vinsFocalScale: 0.62, vinsFx: 0, vinsFy: 0, vinsCx: '', vinsCy: '', vinsBaseline: 0.05,
    vinsMaxCnt: 30, vinsMinDist: 4, vinsFreq: 10, vinsFThreshold: 1.0,
    vinsKeyframeParallax: 13.0, vinsMaxSolverTime: 0.10, vinsMaxIterations: 8,
    imuMaxFrames: 3100, imuSyncTolerance: 3.0, imuSource: 'realsense', replayRealtimeFactor: 2,
    estimateExtrinsic: 0, estimateTd: 0,
    accN: 0.1, gyrN: 0.01, accW: 0.001, gyrW: 0.0001
  },
  vins_stereo_imu_precision_3100: {
    startFrame: 0, vinsMaxFrames: 3100, vinsStride: 1, vinsResizeWidth: 848, vinsFps: 0,
    vinsFocalScale: 0.62, vinsFx: 0, vinsFy: 0, vinsCx: '', vinsCy: '', vinsBaseline: 0.05004,
    vinsMaxCnt: 160, vinsMinDist: 35, vinsFreq: 10, vinsFThreshold: 0.75,
    vinsKeyframeParallax: 10.0, vinsMaxSolverTime: 0.10, vinsMaxIterations: 15,
    imuMaxFrames: 3100, imuSyncTolerance: 3.0, imuSource: 'realsense', replayRealtimeFactor: 2,
    estimateExtrinsic: 0, estimateTd: 0,
    accN: 0.1, gyrN: 0.01, accW: 0.001, gyrW: 0.0001
  },
  vins_klt_reference: {
    startFrame: 0, vinsMaxFrames: 100, vinsStride: 1, vinsResizeWidth: 848, vinsFps: 0,
    vinsFocalScale: 0.62, vinsFx: 0, vinsFy: 0, vinsCx: '', vinsCy: '', vinsBaseline: 0.05,
    vinsMaxCnt: 120, vinsMinDist: 35, vinsFreq: 10, vinsFThreshold: 0.75,
    vinsKeyframeParallax: 12.0, vinsMaxSolverTime: 0.04, vinsMaxIterations: 8,
    imuMaxFrames: 150, imuSyncTolerance: 3.0, imuSource: 'realsense', replayRealtimeFactor: 12,
    estimateExtrinsic: 1, estimateTd: 1,
    accN: 0.1, gyrN: 0.01, accW: 0.001, gyrW: 0.0001
  },
  vins_smoke: {
    startFrame: 0, vinsMaxFrames: 100, vinsStride: 1, vinsResizeWidth: 848, vinsFps: 0,
    vinsFocalScale: 0.62, vinsFx: 0, vinsFy: 0, vinsCx: '', vinsCy: '', vinsBaseline: 0.05,
    vinsMaxCnt: 150, vinsMinDist: 30, vinsFreq: 10, vinsFThreshold: 1.0,
    vinsKeyframeParallax: 10.0, vinsMaxSolverTime: 0.04, vinsMaxIterations: 8
  },
  vins_pool_conservative: {
    startFrame: 0, vinsMaxFrames: 300, vinsStride: 1, vinsResizeWidth: 848, vinsFps: 0,
    vinsFocalScale: 0.62, vinsFx: 0, vinsFy: 0, vinsCx: '', vinsCy: '', vinsBaseline: 0.05,
    vinsMaxCnt: 120, vinsMinDist: 40, vinsFreq: 10, vinsFThreshold: 0.75,
    vinsKeyframeParallax: 12.0, vinsMaxSolverTime: 0.04, vinsMaxIterations: 8
  },
  vins_pool_sensitive: {
    startFrame: 0, vinsMaxFrames: 300, vinsStride: 1, vinsResizeWidth: 848, vinsFps: 0,
    vinsFocalScale: 0.62, vinsFx: 0, vinsFy: 0, vinsCx: '', vinsCy: '', vinsBaseline: 0.05,
    vinsMaxCnt: 220, vinsMinDist: 20, vinsFreq: 10, vinsFThreshold: 1.25,
    vinsKeyframeParallax: 8.0, vinsMaxSolverTime: 0.06, vinsMaxIterations: 10
  },
  vins_full: {
    startFrame: 0, vinsMaxFrames: 0, vinsStride: 1, vinsResizeWidth: 848, vinsFps: 0,
    vinsFocalScale: 0.62, vinsFx: 0, vinsFy: 0, vinsCx: '', vinsCy: '', vinsBaseline: 0.05,
    vinsMaxCnt: 150, vinsMinDist: 30, vinsFreq: 10, vinsFThreshold: 1.0,
    vinsKeyframeParallax: 10.0, vinsMaxSolverTime: 0.04, vinsMaxIterations: 8
  }
};

function applyVinsPreset() {
  const preset = VINS_PRESETS[value('vinsPresetSelect')];
  if (!preset) return;
  Object.entries(preset).forEach(([id, val]) => setValue(id, val));
  updateTrackerInsight();
}

function attachTrackerListeners() {
  [
    'vinsMaxCnt',
    'vinsMinDist',
    'vinsFreq',
    'vinsFThreshold',
    'vinsKeyframeParallax',
    'vinsMaxSolverTime',
    'vinsMaxIterations'
  ].forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.addEventListener('input', updateTrackerInsight);
  });
}

function updateTrackerInsight() {
  const presetId = value('vinsPresetSelect');
  const presetLabel = document.getElementById('vinsPresetSelect').selectedOptions[0]?.textContent || '';
  const maxCnt = value('vinsMaxCnt');
  const minDist = value('vinsMinDist');
  const fThreshold = value('vinsFThreshold');
  const parallax = value('vinsKeyframeParallax');
  const presetNote = presetId === 'vins_stereo_only_presentation_3100'
    ? '이 프리셋은 기존 3100프레임 VINS stereo-only 결과에서 path length 약 97m가 나온 발표용 설정입니다. IMU fusion 결과가 아니라는 점을 분명히 표시해야 합니다.'
    : '초기 MP4 front-end 실험에서 tracking 지속성과 inlier 비율이 안정적이었던 조합입니다.';
  document.getElementById('trackerInsight').innerHTML = `
    <strong><span class="badge">ACTIVE PRESET</span> ${escapeHtml(presetLabel)}</strong>
    ${escapeHtml(presetNote)}
    VINS-Fusion 내부도 feature를 추적하고 geometric outlier를 제거하는 구조라,
    아래 VINS tracker 값은 그 결과를 반영한 보수적 설정으로 맞춰집니다.<br>
    현재 preset: ${escapeHtml(presetLabel)}<br>
    max_cnt=${escapeHtml(maxCnt)}, min_dist=${escapeHtml(minDist)}, F_threshold=${escapeHtml(fThreshold)}, keyframe_parallax=${escapeHtml(parallax)}
  `;
}

function renderFeatureBenchmark(benchmark) {
  const target = document.getElementById('featureBenchmark');
  if (!benchmark || !benchmark.best_by_method || benchmark.best_by_method.length === 0) {
    target.innerHTML = `
      <h2>Feature Tracking Benchmark</h2>
      <p class="lead">아직 비교 가능한 summary가 없습니다. Run VINS 또는 기존 frontend run을 만든 뒤 다시 열면 표가 채워집니다.</p>
    `;
    return;
  }
  const rows = benchmark.best_by_method.map((row) => {
    const recommended = row.method === benchmark.recommended_method;
    return `
      <tr class="${recommended ? 'recommended' : ''}">
        <td class="method">${escapeHtml(methodLabel(row.method))} ${recommended ? '<span class="badge">추천</span>' : ''}</td>
        <td>${pct(row.pose_success_ratio)}</td>
        <td>${pct(row.mean_geometry_inlier_ratio)}</td>
        <td>${num(row.mean_geometry_inliers, 1)}</td>
        <td>${num(row.mean_active_tracks, 1)}</td>
        <td>${row.frames_processed}</td>
        <td class="source">${escapeHtml(row.input_name || row.summary_name || '')}</td>
      </tr>
    `;
  }).join('');
  const rec = benchmark.recommended;
  const recText = rec
    ? `${methodLabel(rec.method)}: pose success ${pct(rec.pose_success_ratio)}, inlier ratio ${pct(rec.mean_geometry_inlier_ratio)}`
    : 'Shi-Tomasi + KLT를 추천 기준으로 사용합니다.';
  target.innerHTML = `
    <h2>Feature Tracking Benchmark</h2>
    <p class="lead">${escapeHtml(recText)}. 이 표는 outputs 아래 summary JSON에서 50프레임 이상 실험만 골라 계산합니다.</p>
    <table>
      <thead>
        <tr>
          <th>Method</th>
          <th>Pose success</th>
          <th>Inlier ratio</th>
          <th>Inliers</th>
          <th>Tracks</th>
          <th>Frames</th>
          <th>Source</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
  `;
}

function vinsParams() {
  return {
    left_video: document.getElementById('inputPath').value,
    right_video: document.getElementById('rightInputPath').value,
    start_frame: intValue('startFrame'),
    max_frames: intValue('vinsMaxFrames'),
    stride: intValue('vinsStride'),
    resize_width: intValue('vinsResizeWidth'),
    fps: floatValue('vinsFps'),
    focal_scale: floatValue('vinsFocalScale'),
    fx: optionalFloat('vinsFx'),
    fy: optionalFloat('vinsFy'),
    cx: optionalFloat('vinsCx'),
    cy: optionalFloat('vinsCy'),
    baseline: floatValue('vinsBaseline'),
    max_cnt: intValue('vinsMaxCnt'),
    min_dist: intValue('vinsMinDist'),
    freq: intValue('vinsFreq'),
    f_threshold: floatValue('vinsFThreshold'),
    max_solver_time: floatValue('vinsMaxSolverTime'),
    max_num_iterations: intValue('vinsMaxIterations'),
    keyframe_parallax: floatValue('vinsKeyframeParallax')
  };
}

function stereoImuParams() {
  return {
    bag_path: value('rosbagPath'),
    start_frame: intValue('startFrame'),
    max_frames: intValue('imuMaxFrames'),
    stride: intValue('vinsStride'),
    sync_tolerance_ms: floatValue('imuSyncTolerance'),
    imu_source: value('imuSource'),
    replay_realtime_factor: floatValue('replayRealtimeFactor'),
    estimate_extrinsic: intValue('estimateExtrinsic'),
    estimate_td: intValue('estimateTd'),
    max_cnt: intValue('vinsMaxCnt'),
    min_dist: intValue('vinsMinDist'),
    freq: intValue('vinsFreq'),
    f_threshold: floatValue('vinsFThreshold'),
    max_solver_time: floatValue('vinsMaxSolverTime'),
    max_num_iterations: intValue('vinsMaxIterations'),
    keyframe_parallax: floatValue('vinsKeyframeParallax'),
    acc_n: floatValue('accN'),
    gyr_n: floatValue('gyrN'),
    acc_w: floatValue('accW'),
    gyr_w: floatValue('gyrW')
  };
}

function ros2StereoImuParams() {
  return {
    ...stereoImuParams(),
    feature_detector: value('ros2FeatureDetector'),
    depth_scale: floatValue('ros2DepthScale'),
    max_features: intValue('ros2MaxFeatures'),
    min_features: intValue('ros2MinFeatures'),
    cell_max_features: intValue('ros2CellMaxFeatures'),
    min_distance: intValue('ros2MinDistance'),
    orb_fast_threshold: intValue('ros2OrbFastThreshold'),
    fast_threshold: intValue('ros2OrbFastThreshold'),
    pnp_threshold: floatValue('ros2PnpThreshold'),
    min_pnp_inliers: 18,
    temporal_ransac: value('ros2TemporalRansac'),
    temporal_ransac_threshold: floatValue('ros2TemporalThreshold'),
    min_temporal_inliers: 18,
    reject_repeated_patches: document.getElementById('ros2RejectRepeatedPatches').checked,
    klt_max_flow_px: floatValue('ros2KltMaxFlowPx'),
    klt_patch_ncc_threshold: floatValue('ros2KltPatchNcc'),
    klt_patch_size: intValue('ros2KltPatchSize'),
    klt_max_accel_px: floatValue('ros2KltMaxAccelPx'),
    klt_min_eigenvalue: floatValue('ros2KltMinEigenvalue'),
    min_track_age_for_pnp: intValue('ros2MinTrackAge'),
    klt_local_flow_check: document.getElementById('ros2KltLocalFlowCheck').checked,
    klt_use_prediction: document.getElementById('ros2KltUsePrediction').checked,
    klt_velocity_check: document.getElementById('ros2KltVelocityCheck').checked,
    max_step_m: floatValue('ros2MaxStepM'),
    max_speed_mps: 1.0,
    max_rotation_rad: 0.45,
    imu_rotation_gate_rad: 0.75,
    translation_smoothing: floatValue('ros2TranslationSmoothing'),
    rate: floatValue('rvizPublishHz')
  };
}

async function makeVinsConfig() {
  setBusy(true, 'Generating VINS dataset/config...');
  try {
    const res = await fetch('/api/vins_config', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(vinsParams())
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'VINS config failed');
    renderVins(data, false);
    setStatus('VINS config ready');
  } catch (err) {
    setStatus(err.message, true);
  } finally {
    setBusy(false);
  }
}

async function buildVinsDocker() {
  setBusy(true, 'Building VINS-Fusion Docker image. This can take several minutes...');
  try {
    const res = await fetch('/api/vins_build', {method: 'POST'});
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Docker build failed');
    setStatus('Docker image ready: underwater-vins-fusion:kinetic');
    document.getElementById('sweepResults').innerHTML =
      `<div class="notes">Docker image built. You can now click Run VINS.</div><pre class="log">${escapeHtml(data.log || '')}</pre>`;
  } catch (err) {
    setStatus(err.message, true);
  } finally {
    setBusy(false);
  }
}

async function runVins() {
  setBusy(true, 'Running VINS-Fusion...');
  try {
    const res = await fetch('/api/vins_run', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(vinsParams())
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'VINS run failed');
    renderVins(data, true);
    setStatus('VINS run done');
  } catch (err) {
    setStatus(err.message, true);
  } finally {
    setBusy(false);
  }
}

async function runStereoImuVins() {
  setBusy(true, 'Exporting ROSBag stereo+IMU and running VINS-Fusion...');
  try {
    const res = await fetch('/api/vins_stereo_imu_run', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(stereoImuParams())
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Stereo+IMU VINS run failed');
    renderVins(data, true);
    setStatus('Stereo+IMU VIO run done');
  } catch (err) {
    setStatus(err.message, true);
  } finally {
    setBusy(false);
  }
}

async function runRos2StereoImuVio() {
  setBusy(true, 'Running ROS2-native stereo+IMU VIO. DVL is reference-only.');
  try {
    const res = await fetch('/api/ros2_stereo_imu_vio_run', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(ros2StereoImuParams())
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'ROS2 Stereo+IMU VIO failed');
    renderRos2Vio(data);
    setStatus(`ROS2 stereo+IMU VIO done: path ${num(data.trajectory_stats.path_length_m, 2)}m, success ${num(data.pose_success_ratio * 100, 1)}%. DVL was not used.`);
  } catch (err) {
    setStatus(err.message, true);
  } finally {
    setBusy(false);
  }
}

async function restartRvizPublisher() {
  setBusy(true, 'Restarting RViz live publisher...');
  try {
    const res = await fetch('/api/rviz_publisher_restart', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        rate: floatValue('rvizPublishHz'),
        csv: value('rvizLiveCsv')
      })
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'RViz publisher restart failed');
    setStatus(`RViz publisher: ${data.rows} poses at ${num(data.rate, 1)} Hz, ${num(data.seconds_per_pass, 1)} s/pass`);
    document.getElementById('sweepResults').innerHTML =
      `<div class="notes">RViz live publisher restarted. CSV=${escapeHtml(data.csv)}, pid=${escapeHtml(data.pid)}</div>`;
  } catch (err) {
    setStatus(err.message, true);
  } finally {
    setBusy(false);
  }
}

async function applyAquaCorrection() {
  setBusy(true, 'Applying AQUA-style metric correction...');
  try {
    const res = await fetch('/api/aqua_metric_correct', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        input_csv: value('aquaInputCsv'),
        output_csv: value('aquaOutputCsv'),
        source: value('aquaMetricSource'),
        target_path_m: floatValue('aquaTargetPathM'),
        max_step_m: floatValue('aquaMaxStepM'),
        rate: floatValue('rvizPublishHz')
      })
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'AQUA correction failed');
    setStatus(`AQUA correction applied: ${num(data.target_path_length_m, 2)}m, scale ${num(data.scale, 3)}`);
    document.getElementById('summary').innerHTML = [
      metric('Corrected rows', data.rows),
      metric('Raw path', `${num(data.raw_path_length_m, 2)} m`),
      metric('Corrected path', `${num(data.target_path_length_m, 2)} m`),
      metric('Scale', num(data.scale, 3)),
      metric('Clamped steps', data.clamped_steps),
      metric('RViz playback', `${num(data.publisher.seconds_per_pass, 1)} s/pass`)
    ].join('');
    document.getElementById('links').innerHTML = [
      linkLine('AQUA corrected CSV', data.output_csv),
      linkLine('Live RViz CSV', data.live_odometry_csv || '')
    ].join('');
    document.getElementById('sweepResults').innerHTML =
      `<div class="notes">AQUA-style correction source=${escapeHtml(data.source)}. RViz live CSV was replaced and publisher restarted.</div>`;
  } catch (err) {
    setStatus(err.message, true);
  } finally {
    setBusy(false);
  }
}

async function fitToDvlOdom() {
  setBusy(true, 'Evaluating stereo+IMU VIO against DVL reference...');
  try {
    const res = await fetch('/api/dvl_fit', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        input_csv: value('aquaInputCsv'),
        output_csv: 'outputs/live/latest_odometry_dvl_fit.csv',
        max_step_grid: value('dvlFitGrid'),
        dvl_clean_max_step_grid: value('dvlCleanMaxStepGrid'),
        dvl_reference_csv: 'outputs/evaluation/dvl_xyz_30s_reference.csv',
        summary_json: 'outputs/evaluation/dvl_xyz_30s_fit_summary.json',
        start_sec: floatValue('dvlStartSec'),
        duration_sec: floatValue('dvlDurationSec'),
        fit_modes: value('dvlFitModes'),
        min_mean_corr: floatValue('dvlMinMeanCorr'),
        max_rmse_ratio: floatValue('dvlMaxRmseRatio'),
        max_error_ratio: floatValue('dvlMaxErrorRatio'),
        rate: floatValue('rvizPublishHz'),
        publish_live: false
      })
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'DVL fit failed');
    renderDvlFit(data);
    const reached = data.reached_min_corr ? 'target reached' : 'target not reached';
    setStatus(`DVL diagnostic only: ${reached}, stage ${data.controller.stage}, RMSE ${num(data.best.rmse_m, 2)}m, corr ${num(data.best.mean_corr, 3)}. Blue VIO path was not changed.`);
  } catch (err) {
    setStatus(err.message, true);
  } finally {
    setBusy(false);
  }
}

async function showDvlOracle() {
  setBusy(true, 'Publishing exact DVL reference path to RViz...');
  try {
    const res = await fetch('/api/dvl_fit', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        input_csv: value('aquaInputCsv'),
        output_csv: 'outputs/live/latest_odometry_dvl_oracle.csv',
        max_step_grid: '1000.0',
        dvl_clean_max_step_grid: '0',
        dvl_reference_csv: 'outputs/evaluation/dvl_xyz_30s_reference.csv',
        summary_json: 'outputs/evaluation/dvl_xyz_30s_oracle_summary.json',
        start_sec: floatValue('dvlStartSec'),
        duration_sec: floatValue('dvlDurationSec'),
        fit_modes: 'oracle',
        min_mean_corr: 1.0,
        max_rmse_ratio: 0.000001,
        max_error_ratio: 0.000001,
        rate: floatValue('rvizPublishHz'),
        publish_live: false
      })
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'DVL oracle failed');
    renderDvlFit(data);
    setStatus(`DVL red reference only: ${data.rows} poses, path ${num(data.dvl_path_length_m, 2)}m. Blue VIO path was not changed.`);
  } catch (err) {
    setStatus(err.message, true);
  } finally {
    setBusy(false);
  }
}

async function fitToSimRcReplay() {
  setBusy(true, 'Fitting VINS trajectory to MuJoCo RC replay reference...');
  try {
    const simBag = value('simRcBagPath');
    const payload = {
      input_csv: value('aquaInputCsv'),
      output_csv: 'outputs/live/latest_odometry_uuv_sim_fit.csv',
      reference_csv: 'outputs/evaluation/uuv_sim_rc_replay_raw_xy_reference.csv',
      sim_bag: simBag === 'latest' ? '' : simBag,
      reference_topic: value('simRcReferenceTopic'),
      time_map: value('simRcTimeMap'),
      offset_target: value('simRcOffsetTarget'),
      metric_axes: value('simRcMetricAxes'),
      selection_mode: value('simRcSelectionMode'),
      output_z_mode: 'zero',
      time_offset_grid: value('simRcTimeOffsetGrid'),
      max_step_grid: value('simRcFitGrid'),
      fit_modes: value('simRcFitModes'),
      min_mean_corr: floatValue('dvlMinMeanCorr'),
      max_rmse_ratio: floatValue('dvlMaxRmseRatio'),
      max_error_ratio: floatValue('dvlMaxErrorRatio'),
      rate: floatValue('rvizPublishHz')
    };
    const res = await fetch('/api/sim_rc_replay_fit', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'MuJoCo RC replay fit failed');
    renderSimRcFit(data);
    const reached = data.reached_min_corr ? 'target reached' : 'target not reached';
    setStatus(`MuJoCo RC replay fit: ${reached}, stage ${data.controller.stage}, RMSE ${num(data.best.rmse_m, 2)}m, corr ${num(data.best.mean_corr, 3)}`);
  } catch (err) {
    setStatus(err.message, true);
  } finally {
    setBusy(false);
  }
}

async function publishFixedSimRcReference() {
  setBusy(true, 'Publishing raw MuJoCo sim odom XY reference as a fixed red RViz path...');
  try {
    const simBag = value('simRcBagPath');
    const res = await fetch('/api/sim_rc_reference_publish', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        input_csv: value('aquaInputCsv'),
        reference_csv: 'outputs/evaluation/uuv_sim_rc_replay_raw_xy_reference.csv',
        sim_bag: simBag === 'latest' ? '' : simBag,
        reference_topic: value('simRcReferenceTopic'),
        time_map: value('simRcTimeMap'),
        offset_target: 'reference',
        metric_axes: value('simRcMetricAxes'),
        output_z_mode: 'zero',
        time_offset_sec: 0,
        rate: 1.0
      })
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Raw MuJoCo reference publish failed');
    renderSimRcReference(data);
    setStatus(`Raw sim odom reference is fixed in RViz: ${num(data.reference_path_length_m, 2)}m, ${data.rows} poses, red path`);
  } catch (err) {
    setStatus(err.message, true);
  } finally {
    setBusy(false);
  }
}

async function showSimRcOracle() {
  setBusy(true, 'Publishing MuJoCo RC replay reference path to RViz...');
  try {
    const simBag = value('simRcBagPath');
    const res = await fetch('/api/sim_rc_replay_fit', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        input_csv: value('aquaInputCsv'),
        output_csv: 'outputs/live/latest_odometry_uuv_sim_oracle.csv',
        reference_csv: 'outputs/evaluation/uuv_sim_rc_replay_raw_xy_reference.csv',
        sim_bag: simBag === 'latest' ? '' : simBag,
        reference_topic: value('simRcReferenceTopic'),
        time_map: value('simRcTimeMap'),
        offset_target: 'reference',
        metric_axes: value('simRcMetricAxes'),
        selection_mode: 'max-corr',
        output_z_mode: 'zero',
        time_offset_grid: '0',
        max_step_grid: '1000.0',
        fit_modes: 'oracle',
        min_mean_corr: 1.0,
        max_rmse_ratio: 0.000001,
        max_error_ratio: 0.000001,
        rate: floatValue('rvizPublishHz')
      })
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'MuJoCo RC replay oracle failed');
    renderSimRcFit(data);
    setStatus(`MuJoCo RC reference: ${num(data.reference_path_length_m, 2)}m, ${data.rows} poses`);
  } catch (err) {
    setStatus(err.message, true);
  } finally {
    setBusy(false);
  }
}

function renderDvlFit(data) {
  document.getElementById('summary').innerHTML = [
    metric('Rows', data.rows),
    metric('Segment', `${num(data.start_sec, 1)}s + ${num(data.duration_sec, 1)}s`),
    metric('DVL path', `${num(data.dvl_path_length_m, 2)} m`),
    metric('Raw VINS path', `${num(data.raw_vins_path_length_m, 2)} m`),
    metric('Target reached', data.reached_min_corr ? 'YES' : 'NO'),
    metric('Controller stage', data.controller ? data.controller.stage : ''),
    metric('Best fit mode', data.best.fit_mode),
    metric('Best RMSE', `${num(data.best.rmse_m, 2)} m`),
    metric('Mean corr', num(data.best.mean_corr, 3)),
    metric('RMSE/path', num(data.best.rmse_ratio, 3)),
    metric('MaxErr/path', num(data.best.max_error_ratio, 3)),
    metric('DVL clean max_step', `${num(data.best.dvl_clean_max_step_m, 2)} m`),
    metric('Best max_step', `${num(data.best.max_step_m, 2)} m`)
  ].join('');
  document.getElementById('links').innerHTML = [
    linkLine('DVL diagnostic output CSV', data.output_csv),
    linkLine('DVL clean reference CSV', data.dvl_reference_csv),
    linkLine('Live RViz CSV', data.live_odometry_csv || ''),
    linkLine('Red DVL reference log', data.reference_publisher ? data.reference_publisher.log : '')
  ].join('');
  const rows = (data.top_candidates || []).slice(0, 8).map((row, index) => `
    <tr class="${index === 0 ? 'recommended' : ''}">
      <td>${escapeHtml(row.fit_mode || '')}</td>
      <td>${num(row.dvl_clean_max_step_m, 2)}</td>
      <td>${num(row.max_step_m, 2)}</td>
      <td>${num(row.rmse_m, 3)} m</td>
      <td>${num(row.mean_corr, 3)}</td>
      <td>${row.control_pass ? 'PASS' : 'FAIL'}</td>
      <td>${escapeHtml((row.control_failures || []).join(', '))}</td>
      <td>${num(row.corr_x, 3)}</td>
      <td>${num(row.corr_y, 3)}</td>
      <td>${num(row.corr_z, 3)}</td>
      <td>${row.clamped_steps}</td>
    </tr>
  `).join('');
  document.getElementById('sweepResults').innerHTML = `
    <h2>VIO vs DVL Diagnostic</h2>
    <p class="lead">DVL is reference-only here. It is not fed back into the stereo+IMU VIO estimate. ${escapeHtml(data.controller ? data.controller.reason : '')} ${escapeHtml(data.controller && data.controller.warning ? data.controller.warning : '')}</p>
    <table>
      <thead>
        <tr>
          <th>mode</th><th>DVL clean</th><th>VINS max_step</th><th>RMSE</th><th>mean corr</th><th>gate</th><th>failures</th><th>corr_x</th><th>corr_y</th><th>corr_z</th><th>clamped</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
  `;
}

function renderSimRcFit(data) {
  document.getElementById('summary').innerHTML = [
    metric('Rows', data.rows),
    metric('Sim reference path', `${num(data.reference_path_length_m, 2)} m`),
    metric('Raw VINS path', `${num(data.raw_vins_path_length_m, 2)} m`),
    metric('Metric axes', data.metric_axes || ''),
    metric('Selection', data.selection_mode || ''),
    metric('Offset target', data.offset_target || ''),
    metric('Time offset', `${num(data.selected_time_offset_sec, 1)} s`),
    metric('Sim replay phase', `${num(data.sim_phase_duration_sec, 1)} s`),
    metric('Reference samples', data.sim_reference_samples),
    metric('Target reached', data.reached_min_corr ? 'YES' : 'NO'),
    metric('Controller stage', data.controller ? data.controller.stage : ''),
    metric('Best fit mode', data.best.fit_mode),
    metric('Best RMSE', `${num(data.best.rmse_m, 2)} m`),
    metric('Mean corr', num(data.best.mean_corr, 3))
  ].join('');
  document.getElementById('links').innerHTML = [
    linkLine('MuJoCo-fitted live CSV', data.output_csv),
    linkLine('MuJoCo reference CSV', data.reference_csv),
    linkLine('Sim bag', data.sim_bag),
    linkLine('Live RViz CSV', data.live_odometry_csv || ''),
    linkLine('Red reference publisher log', data.reference_publisher ? data.reference_publisher.log : '')
  ].join('');
  const rows = (data.top_candidates || []).slice(0, 8).map((row, index) => `
    <tr class="${index === 0 ? 'recommended' : ''}">
      <td>${escapeHtml(row.fit_mode || '')}</td>
      <td>${num(row.max_step_m, 2)}</td>
      <td>${num(row.rmse_m, 3)} m</td>
      <td>${num(row.mean_corr, 3)}</td>
      <td>${row.control_pass ? 'PASS' : 'FAIL'}</td>
      <td>${escapeHtml((row.control_failures || []).join(', '))}</td>
      <td>${num(row.corr_x, 3)}</td>
      <td>${num(row.corr_y, 3)}</td>
      <td>${num(row.corr_z, 3)}</td>
      <td>${row.clamped_steps}</td>
    </tr>
  `).join('');
  document.getElementById('sweepResults').innerHTML = `
    <h2>MuJoCo RC Replay Fit Result</h2>
    <p class="lead">${escapeHtml(data.note || '')}</p>
    <p class="lead">${escapeHtml(data.controller ? data.controller.reason : '')} ${escapeHtml(data.controller && data.controller.warning ? data.controller.warning : '')}</p>
    <table>
      <thead>
        <tr>
          <th>mode</th><th>VINS max_step</th><th>RMSE</th><th>mean corr</th><th>gate</th><th>failures</th><th>corr_x</th><th>corr_y</th><th>corr_z</th><th>clamped</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
  `;
}

function renderSimRcReference(data) {
  document.getElementById('preview').innerHTML =
    '<span style="color:#cbd5e1">Raw sim odom 기준선은 RViz에서 빨간색 /sim_odom_reference_path로 고정 표시됩니다.</span>';
  document.getElementById('summary').innerHTML = [
    metric('Rows', data.rows),
    metric('Raw sim ref path', `${num(data.reference_path_length_m, 2)} m`),
    metric('Metric axes', data.metric_axes || ''),
    metric('Time offset', `${num(data.selected_time_offset_sec, 1)} s`),
    metric('Sim replay phase', `${num(data.sim_phase_duration_sec, 1)} s`),
    metric('Reference samples', data.sim_reference_samples),
    metric('RViz topic', data.publisher ? data.publisher.path_topic : '/sim_odom_reference_path')
  ].join('');
  document.getElementById('links').innerHTML = [
    linkLine('Raw MuJoCo reference CSV', data.reference_csv),
    linkLine('Sim bag', data.sim_bag),
    linkLine('Red reference publisher log', data.publisher ? data.publisher.log : '')
  ].join('');
  document.getElementById('sweepResults').innerHTML = `
    <h2>Raw Sim Odom Reference</h2>
    <p class="lead">
      이 빨간 경로는 fit 후보 선택과 무관하게 time_offset=0으로 고정한 MuJoCo sim odom XY 기준입니다.
      파란 /local_path는 Fit 버튼으로 갱신되는 VINS 보정 경로입니다.
    </p>
  `;
}

function updateSourcePreview() {
  const path = document.getElementById('inputPath').value;
  if (!path) return;
  const url = `/file?path=${encodeURIComponent(path)}&t=${Date.now()}`;
  document.getElementById('sourcePreview').innerHTML =
    `<video id="sourceVideo" controls autoplay muted loop playsinline src="${url}"></video>`;
  document.getElementById('sourceInfo').textContent = path.split('/').pop() || '';
}

function renderAccuracyBenchmark(benchmark) {
  const target = document.getElementById('accuracyBenchmark');
  if (!benchmark || !benchmark.rows || benchmark.rows.length === 0) {
    target.innerHTML = `
      <h2>Trajectory Accuracy Benchmark</h2>
      <p class="lead">아직 accuracy CSV가 없습니다. outputs/evaluation/trajectory_accuracy_all_runs.csv를 생성하면 표시됩니다.</p>
    `;
    return;
  }
  const rows = benchmark.rows.slice(0, 8).map((row, index) => `
    <tr class="${index === 0 ? 'recommended' : ''}">
      <td class="method">${escapeHtml(methodLabel(row.method))} ${index === 0 ? '<span class="badge">ATE best</span>' : ''}</td>
      <td>${num(row.sim3_rmse, 3)} m</td>
      <td>${num(row.duration_sec, 1)} s</td>
      <td>${num(row.raw_rmse, 2)} m</td>
      <td>${num(row.sim3_scale, 3)}</td>
      <td class="source">${escapeHtml(row.video || '')}</td>
    </tr>
  `).join('');
  target.innerHTML = `
    <h2>Trajectory Accuracy Benchmark</h2>
    <p class="lead">
      기준: uuv_sim rosbag /odometry/filtered. Sim3 ATE는 좌표축/scale을 맞춘 뒤의 trajectory shape error라,
      MP4-only pseudo VO 비교에 더 공정합니다. 이 기준에서는 KLT가 항상 1등은 아닙니다.
    </p>
    <table>
      <thead>
        <tr>
          <th>Method</th>
          <th>Sim3 ATE</th>
          <th>Duration</th>
          <th>Raw RMSE</th>
          <th>Scale</th>
          <th>Video</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
  `;
}

function renderVins(data, hasRun) {
  document.getElementById('preview').innerHTML =
    '<span style="color:#cbd5e1">VINS-Fusion 결과는 RViz local path로 확인합니다.</span>';
  document.getElementById('summary').innerHTML = [
    metric('Requested max_frames', data.requested_max_frames ?? data.metadata.requested_max_frames ?? data.metadata.frames),
    metric('Exported image pairs', data.metadata.frames),
    metric('FPS', num(data.metadata.fps, 2)),
    metric('Size', `${data.metadata.width} x ${data.metadata.height}`),
    metric('Duration', `${num(data.metadata.duration_sec, 1)} s`),
    metric('VINS poses', poseCoverageText(data)),
    metric('Start', data.metadata.start_frame),
    metric('Stride', data.metadata.stride)
  ].join('');
  document.getElementById('links').innerHTML = [
    pathLine('VINS dataset', data.dataset_dir),
    linkLine('VINS config', data.config_file),
    pathLine('VINS output dir', data.output_dir),
    linkLine('RViz CSV', data.odometry_csv || ''),
    linkLine('Live RViz CSV', data.live_odometry_csv || '')
  ].join('');
  const stats = data.trajectory_stats
    ? ` path=${num(data.trajectory_stats.path_length_m, 2)}m, max_step=${num(data.trajectory_stats.max_step_m, 2)}m, max_speed=${num(data.trajectory_stats.max_speed_mps, 2)}m/s`
    : '';
  const frameNote = frameUsageNote(data);
  const message = hasRun
    ? (data.trajectory_valid === false
      ? `VINS 결과가 발산하거나 일부만 생성되어 RViz live CSV로 복사하지 않았습니다. ${escapeHtml(data.trajectory_validation || '')}${escapeHtml(stats)}${frameNote}`
      : `RViz live publisher가 켜져 있으면 방금 VINS CSV로 바로 갱신됩니다.${escapeHtml(stats)}${frameNote}`)
    : 'Config만 생성했습니다. Run VINS를 누르면 Docker에서 VINS-Fusion을 실행합니다.';
  const log = data.vins_log ? `<pre class="log">${escapeHtml(data.vins_log)}</pre>` : '';
  document.getElementById('sweepResults').innerHTML = `<div class="notes">${message}</div>${log}`;
}

function renderRos2Vio(data) {
  document.getElementById('preview').innerHTML =
    '<span style="color:#cbd5e1">ROS2-native VIO 결과는 RViz 파란 /local_path로 확인합니다.</span>';
  document.getElementById('summary').innerHTML = [
    metric('Estimator input', 'stereo + IMU only'),
    metric('Reference input', 'none'),
    metric('Frames', data.metadata.frames),
    metric('Duration', `${num(data.metadata.duration_sec, 1)} s`),
    metric('Path length', `${num(data.trajectory_stats.path_length_m, 2)} m`),
    metric('Max step', `${num(data.trajectory_stats.max_step_m, 3)} m`),
    metric('Pose success', `${num(data.pose_success_ratio * 100, 1)} %`),
    metric('Mean PnP inliers', num(data.tracking_stats.mean_pnp_inliers, 1))
  ].join('');
  document.getElementById('links').innerHTML = [
    pathLine('ROS2 stereo+IMU dataset', data.dataset_dir),
    linkLine('ROS2 VIO CSV', data.odometry_csv || data.output_csv || ''),
    linkLine('Debug CSV', data.debug_csv || ''),
    linkLine('Summary JSON', data.summary_json || ''),
    linkLine('Live RViz CSV', data.live_odometry_csv || ''),
    linkLine('Blue publisher log', data.publisher ? data.publisher.log : '')
  ].join('');
  const notes = Object.entries(data.note_counts || {})
    .map(([key, value]) => `<tr><td>${escapeHtml(key)}</td><td>${escapeHtml(value)}</td></tr>`)
    .join('');
  document.getElementById('sweepResults').innerHTML = `
    <div class="notes">
      ROS2-native path: stereo triangulation + KLT tracking + PnP RANSAC + IMU rotation gate + physical motion gate.
      DVL is not fed into the estimator; red DVL path remains a visual reference only.
      ${escapeHtml(data.trajectory_validation || '')}
    </div>
    <table>
      <thead><tr><th>state</th><th>frames</th></tr></thead>
      <tbody>${notes}</tbody>
    </table>
  `;
}

function metric(label, value) {
  return `<div class="metric"><span>${label}</span><strong>${escapeHtml(value)}</strong></div>`;
}
function poseCoverageText(data) {
  const poses = Number(data.pose_count ?? 0);
  const frames = Number(data.metadata?.frames ?? 0);
  if (!frames) return poses;
  return `${poses} / ${frames} (${num(poses * 100 / frames, 1)}%)`;
}
function frameUsageNote(data) {
  const requested = Number(data.requested_max_frames ?? data.metadata?.requested_max_frames ?? 0);
  const exported = Number(data.metadata?.frames ?? 0);
  const poses = Number(data.pose_count ?? 0);
  const notes = [];
  if (requested > 0 && exported > 0 && exported < requested) {
    notes.push(`requested ${requested}, exported ${exported}`);
  }
  if (exported > 0 && poses > 0 && poses < exported * 0.9) {
    notes.push(`VINS poses ${poses}/${exported}`);
  }
  return notes.length ? ` frame_check=${escapeHtml(notes.join(', '))}` : '';
}
function linkLine(label, path) {
  if (!path) return '';
  const url = `/file?path=${encodeURIComponent(path)}`;
  return `<div>${label}: <a href="${url}" target="_blank">${escapeHtml(path)}</a></div>`;
}
function pathLine(label, path) {
  if (!path) return '';
  return `<div>${label}: ${escapeHtml(path)}</div>`;
}
function value(id) { return document.getElementById(id).value; }
function setValue(id, v) { document.getElementById(id).value = v; }
function intValue(id) { return parseInt(value(id), 10); }
function floatValue(id) { return parseFloat(value(id)); }
function optionalFloat(id) {
  const text = value(id).trim();
  return text === '' ? null : parseFloat(text);
}
function num(v, digits) {
  if (v === null || v === undefined || Number.isNaN(Number(v))) return 'n/a';
  return Number(v).toFixed(digits);
}
function pct(v) {
  const n = Number(v);
  return Number.isFinite(n) ? `${(n * 100).toFixed(1)}%` : '-';
}
function methodLabel(method) {
  return {
    shi_tomasi_klt: 'Shi-Tomasi + KLT',
    sift_match: 'SIFT matching',
    akaze_match: 'AKAZE matching',
    orb_match: 'ORB matching'
  }[method] || method;
}
function setBusy(busy, text) {
  setDisabled('vinsBuildBtn', busy);
  setDisabled('vinsConfigBtn', busy);
  setDisabled('vinsRunBtn', busy);
  setDisabled('ros2VioRunBtn', busy);
  setDisabled('vinsImuRunBtn', busy);
  setDisabled('aquaCorrectBtn', busy);
  setDisabled('dvlFitBtn', busy);
  setDisabled('dvlOracleBtn', busy);
  setDisabled('simRcFitBtn', busy);
  setDisabled('simRcOracleBtn', busy);
  setDisabled('rvizPublisherBtn', busy);
  if (text) setStatus(text);
}
function setDisabled(id, disabled) {
  const el = document.getElementById(id);
  if (el) el.disabled = disabled;
}
function setStatus(text, error=false) {
  const el = document.getElementById('status');
  el.textContent = text;
  el.className = error ? 'status error' : 'status';
}
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}
init().catch((err) => setStatus(err.message, true));
</script>
</body>
</html>
"""


class WebGuiHandler(BaseHTTPRequestHandler):
    server_version = "UnderwaterVinsFusionWebGui/1.0"

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/":
            self._send_html(INDEX_HTML)
        elif parsed.path == "/api/defaults":
            self._send_json(
                {
                    "videos": _list_videos(),
                    "feature_benchmark": _feature_benchmark(),
                    "accuracy_benchmark": _trajectory_accuracy_benchmark(),
                }
            )
        elif parsed.path == "/file":
            params = urllib.parse.parse_qs(parsed.query)
            path_value = params.get("path", [""])[0]
            self._send_file(Path(path_value))
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        try:
            payload = self._read_json()
            if self.path == "/api/vins_config":
                self._send_json(make_vins_config(payload, run_vins=False))
            elif self.path == "/api/vins_run":
                self._send_json(make_vins_config(payload, run_vins=True))
            elif self.path == "/api/vins_build":
                self._send_json(build_vins_docker())
            elif self.path == "/api/vins_stereo_imu_run":
                self._send_json(run_stereo_imu_vins(payload))
            elif self.path == "/api/ros2_stereo_imu_vio_run":
                self._send_json(run_ros2_stereo_imu_vio(payload))
            elif self.path == "/api/rviz_publisher_restart":
                self._send_json(restart_rviz_publisher(payload))
            elif self.path == "/api/aqua_metric_correct":
                self._send_json(apply_aqua_metric_correction(payload))
            elif self.path == "/api/dvl_fit":
                self._send_json(fit_to_dvl_odom(payload))
            elif self.path == "/api/sim_rc_replay_fit":
                self._send_json(fit_to_sim_rc_replay(payload))
            elif self.path == "/api/sim_rc_reference_publish":
                self._send_json(publish_sim_rc_reference(payload))
            else:
                self.send_error(404)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=500)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[web_gui] {self.address_string()} - {format % args}")

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("JSON payload must be an object")
        return data

    def _send_html(self, text: str) -> None:
        body = text.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, value: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path) -> None:
        resolved = path.resolve()
        if not _is_under_project(resolved) or not resolved.exists() or not resolved.is_file():
            self.send_error(404)
            return
        content_type = _content_type(resolved)
        file_size = resolved.stat().st_size
        range_header = self.headers.get("Range")
        if range_header:
            start, end = _parse_range_header(range_header, file_size)
            with resolved.open("rb") as file:
                file.seek(start)
                data = file.read(end - start + 1)
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        data = resolved.read_bytes()
        self.send_response(200)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def make_vins_config(payload: dict[str, Any], run_vins: bool) -> dict[str, Any]:
    left_video = _require_project_file(payload["left_video"])
    right_video = _require_project_file(payload["right_video"])
    if left_video == right_video:
        raise ValueError("VINS stereo needs different left/right MP4 files.")

    timestamp = time.strftime("run_%Y%m%d_%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}"
    output_root = PROJECT_ROOT / "outputs" / "vins_fusion" / "gui_runs" / timestamp
    dataset_dir = output_root / "underwater_stereo_kitti"
    config_dir = output_root / "config"
    output_dir = output_root / "vins_output"
    odometry_csv = output_root / "vins_odometry.csv"

    export_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "export_vins_kitti_stereo.py"),
        "--left-video",
        str(left_video),
        "--right-video",
        str(right_video),
        "--output-dir",
        str(dataset_dir),
        "--max-frames",
        str(int(payload["max_frames"])),
        "--start-frame",
        str(int(payload.get("start_frame", 0))),
        "--stride",
        str(int(payload.get("stride", 1))),
        "--resize-width",
        str(int(payload["resize_width"])),
        "--force",
    ]
    fps = _optional_float(payload.get("fps"))
    if fps and fps > 0:
        export_cmd.extend(["--fps", str(fps)])
    export_stdout = _run_command(export_cmd, timeout=180)
    metadata = json.loads(export_stdout)

    fx = _optional_float(payload.get("fx"))
    fy = _optional_float(payload.get("fy"))
    cx = _optional_float(payload.get("cx"))
    cy = _optional_float(payload.get("cy"))
    config_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "vins_make_stereo_config.py"),
        "--metadata",
        str(dataset_dir / "metadata.json"),
        "--config-dir",
        str(config_dir),
        "--output-dir",
        str(output_dir),
        "--focal-scale",
        str(float(payload["focal_scale"])),
        "--baseline",
        str(float(payload["baseline"])),
        "--max-cnt",
        str(int(payload["max_cnt"])),
        "--min-dist",
        str(int(payload["min_dist"])),
        "--freq",
        str(int(payload.get("freq", 10))),
        "--f-threshold",
        str(float(payload["f_threshold"])),
        "--max-solver-time",
        str(float(payload["max_solver_time"])),
        "--max-num-iterations",
        str(int(payload["max_num_iterations"])),
        "--keyframe-parallax",
        str(float(payload["keyframe_parallax"])),
    ]
    if fx and fx > 0:
        config_cmd.extend(["--fx", str(fx)])
    if fy and fy > 0:
        config_cmd.extend(["--fy", str(fy)])
    if cx is not None:
        config_cmd.extend(["--cx", str(cx)])
    if cy is not None:
        config_cmd.extend(["--cy", str(cy)])
    config_file = Path(_run_command(config_cmd, timeout=60).strip()).resolve()

    result: dict[str, Any] = {
        "requested_max_frames": int(payload["max_frames"]),
        "metadata": metadata,
        "dataset_dir": str(dataset_dir),
        "config_file": str(config_file),
        "output_dir": str(output_dir),
        "odometry_csv": "",
        "live_odometry_csv": "",
        "pose_count": 0,
    }
    if not run_vins:
        return result

    docker_cmd = [
        str(PROJECT_ROOT / "scripts" / "vins_docker_run_stereo_kitti.sh"),
        str(dataset_dir),
        str(config_file),
    ]
    result["vins_log"] = _tail_lines(_run_command(docker_cmd, timeout=900), 80)

    convert_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "convert_vins_vio_to_csv.py"),
        "--vio-txt",
        str(output_dir / "vio.txt"),
        "--times-txt",
        str(dataset_dir / "times.txt"),
        "--output-csv",
        str(odometry_csv),
    ]
    _run_command(convert_cmd, timeout=60)
    stats = _trajectory_sanity_stats(odometry_csv)
    valid, reason = _is_sane_vio_result(stats, metadata)
    result.update({"trajectory_stats": stats, "trajectory_valid": valid, "trajectory_validation": reason})
    if not valid:
        result["odometry_csv"] = str(odometry_csv)
        result["pose_count"] = max(0, sum(1 for _ in odometry_csv.open(encoding="utf-8")) - 1)
        result["vins_log"] = (result.get("vins_log", "") + "\n\nRejected trajectory: " + reason).strip()
        return result
    result["odometry_csv"] = str(odometry_csv)
    result["pose_count"] = max(0, sum(1 for _ in odometry_csv.open(encoding="utf-8")) - 1)

    live_dir = PROJECT_ROOT / "outputs" / "live"
    live_dir.mkdir(parents=True, exist_ok=True)
    live_odometry = live_dir / "latest_odometry.csv"
    _atomic_copy(odometry_csv, live_odometry)
    result["live_odometry_csv"] = str(live_odometry)
    return result


def build_vins_docker() -> dict[str, Any]:
    if not _docker_available():
        raise RuntimeError(
            "Docker daemon is not running. Open Docker Desktop first, wait until it says Docker is running, then retry."
        )
    output = _run_command([str(PROJECT_ROOT / "scripts" / "vins_docker_build.sh")], timeout=1800)
    return {
        "image": "underwater-vins-fusion:kinetic",
        "image_exists": _docker_image_exists(),
        "log": _tail_lines(output, 120),
    }


def restart_rviz_publisher(payload: dict[str, Any]) -> dict[str, Any]:
    rate = float(payload.get("rate", 60.0))
    if rate <= 0:
        raise ValueError("RViz publish rate must be positive.")
    csv_path = _require_project_file(str(payload.get("csv", "outputs/live/latest_odometry.csv")))
    if not csv_path.exists() or not csv_path.is_file():
        raise ValueError(f"RViz live CSV does not exist: {csv_path}")
    coordinate_frame = str(payload.get("coordinate_frame", "ros"))
    if coordinate_frame not in {"ros", "opencv", "raw", "flip-y"}:
        raise ValueError(f"Unsupported RViz publisher coordinate frame: {coordinate_frame}")

    for pid in _ros2_publisher_pids():
        try:
            os.kill(pid, 15)
        except ProcessLookupError:
            pass
    time.sleep(0.5)
    for pid in _ros2_publisher_pids():
        try:
            os.kill(pid, 9)
        except ProcessLookupError:
            pass

    log_path = Path("/tmp/underwater_vio_live_publisher.log")
    log_file = log_path.open("w", encoding="utf-8")
    command = (
        f"cd {shlex.quote(str(PROJECT_ROOT))} && "
        "set +u && "
        "source /Users/kanghyunmin/miniconda3/bin/activate ros2_h311 && "
        "export ROS_LOCALHOST_ONLY=1 && "
        "set -u && "
        f"exec python scripts/ros2_publish_odometry.py "
        f"--csv {shlex.quote(str(csv_path))} "
        f"--rate {rate} "
        f"--loop --coordinate-frame {shlex.quote(coordinate_frame)} --watch"
    )
    process = subprocess.Popen(
        ["/bin/bash", "-lc", command],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        cwd=PROJECT_ROOT,
        close_fds=True,
    )
    log_file.close()
    rows = _count_csv_rows(csv_path)
    return {
        "pid": process.pid,
        "csv": str(csv_path),
        "rate": rate,
        "rows": rows,
        "seconds_per_pass": rows / rate if rows else 0.0,
        "coordinate_frame": coordinate_frame,
        "log": str(log_path),
    }


def restart_vins_dvl_overlay_publisher(payload: dict[str, Any]) -> dict[str, Any]:
    vins_csv = _require_project_file(str(payload.get("vins_csv", "outputs/live/latest_odometry_vins_pure_stereo_imu.csv")))
    dvl_csv = _require_project_file(str(payload.get("dvl_csv", "outputs/evaluation/dvl_xyz_30s_reference.csv")))
    if not vins_csv.exists() or not vins_csv.is_file():
        raise ValueError(f"VINS overlay CSV does not exist: {vins_csv}")
    if not dvl_csv.exists() or not dvl_csv.is_file():
        raise ValueError(f"DVL reference CSV does not exist: {dvl_csv}")

    for pid in _path_overlay_publisher_pids():
        try:
            os.kill(pid, 15)
        except ProcessLookupError:
            pass
    time.sleep(0.5)
    for pid in _path_overlay_publisher_pids():
        try:
            os.kill(pid, 9)
        except ProcessLookupError:
            pass

    log_path = PROJECT_ROOT / "outputs" / "logs" / "path_overlay_publisher.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", encoding="utf-8")
    command = (
        f"cd {shlex.quote(str(PROJECT_ROOT))} && "
        "set +u && "
        "source /Users/kanghyunmin/miniconda3/bin/activate ros2_h311 && "
        "export ROS_LOCALHOST_ONLY=1 && "
        "set -u && "
        "exec python scripts/ros2_publish_path_overlay.py "
        f"--vins-csv {shlex.quote(str(vins_csv))} "
        f"--dvl-csv {shlex.quote(str(dvl_csv))} "
        "--vins-coordinate-frame ros --dvl-coordinate-frame flip-y "
        "--vins-orientation-source tangent --dvl-orientation-source tangent "
        "--success-only-vins --crop-reference-to-vins-time --sample-reference-at-vins-times"
    )
    process = subprocess.Popen(
        ["/bin/bash", "-lc", command],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        cwd=PROJECT_ROOT,
        close_fds=True,
    )
    log_file.close()
    return {
        "pid": process.pid,
        "vins_csv": str(vins_csv),
        "dvl_csv": str(dvl_csv),
        "vins_rows": _count_csv_rows(vins_csv),
        "dvl_rows": _count_csv_rows(dvl_csv),
        "log": str(log_path),
        "dvl_usage": "reference_overlay_only_not_estimator_input",
    }


def apply_aqua_metric_correction(payload: dict[str, Any]) -> dict[str, Any]:
    input_csv = _require_project_file(payload.get("input_csv"))
    output_csv = _project_output_path(payload.get("output_csv", "outputs/live/latest_odometry_aqua_corrected.csv"))
    source = str(payload.get("source", "reference"))
    if source not in {"reference", "dvl", "target"}:
        raise ValueError(f"Unsupported AQUA correction source: {source}")

    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "aqua_metric_correct_odometry.py"),
        "--input-csv",
        str(input_csv),
        "--output-csv",
        str(output_csv),
        "--source",
        source,
        "--target-path-m",
        str(float(payload.get("target_path_m", 100.0))),
        "--max-step-m",
        str(float(payload.get("max_step_m", 0.3))),
    ]
    if source == "dvl":
        shell_command = "set +u; source /Users/kanghyunmin/miniconda3/bin/activate ros2_h311; " + " ".join(
            shlex.quote(part) for part in command
        )
        stdout = _run_command(["/bin/bash", "-lc", shell_command], timeout=180)
    else:
        stdout = _run_command(command, timeout=60)
    result = json.loads(stdout)

    live_dir = PROJECT_ROOT / "outputs" / "live"
    live_dir.mkdir(parents=True, exist_ok=True)
    live_odometry = live_dir / "latest_odometry.csv"
    _atomic_copy(output_csv, live_odometry)
    publisher = restart_rviz_publisher(
        {
            "rate": float(payload.get("rate", 60.0)),
            "csv": str(live_odometry),
            "coordinate_frame": "flip-y",
        }
    )
    result["live_odometry_csv"] = str(live_odometry)
    result["publisher"] = publisher
    if bool(payload.get("publish_reference", True)):
        result["reference_publisher"] = restart_sim_reference_publisher(
            {
                "csv": str(reference_csv),
                "rate": float(payload.get("reference_rate", 1.0)),
                "line_width": float(payload.get("reference_line_width", 0.08)),
            }
        )
    return result


def fit_to_dvl_odom(payload: dict[str, Any]) -> dict[str, Any]:
    input_csv = _require_project_file(payload.get("input_csv"))
    output_csv = _project_output_path(payload.get("output_csv", "outputs/live/latest_odometry_dvl_fit.csv"))
    dvl_reference_csv = _project_output_path(
        payload.get("dvl_reference_csv", "outputs/evaluation/dvl_position_clean_reference.csv")
    )
    summary_json = _project_output_path(payload.get("summary_json", "outputs/evaluation/dvl_fit_summary.json"))
    overlay_setup = PROJECT_ROOT / "ros2_overlay_ws" / "install" / "setup.bash"
    if not overlay_setup.exists():
        raise RuntimeError(
            "Missing dvl_msgs ROS2 overlay. Build it first with: "
            "cd ros2_overlay_ws && colcon build --packages-select dvl_msgs"
        )

    command = [
        "python",
        str(PROJECT_ROOT / "scripts" / "dvl_fit_vins_odometry.py"),
        "--input-csv",
        str(input_csv),
        "--output-csv",
        str(output_csv),
        "--summary-json",
        str(summary_json),
        "--dvl-reference-csv",
        str(dvl_reference_csv),
        "--dvl-mode",
        "position",
        "--start-sec",
        str(float(payload.get("start_sec", 0.0))),
        "--duration-sec",
        str(float(payload.get("duration_sec", 0.0))),
        "--dvl-clean-max-step-grid",
        str(payload.get("dvl_clean_max_step_grid", "0.05,0.08,0.10,0.12,0.15,0.20,0.25,0.30,0.40,0.50,0")),
        "--max-step-grid",
        str(payload.get("max_step_grid", "0.05,0.08,0.10,0.15,0.20,0.30,0.40,0.50,0.75,1.0,1.5,2.0,3.0,5.0,1000.0")),
        "--fit-modes",
        str(payload.get("fit_modes", "rigid,sim3,affine")),
        "--min-mean-corr",
        str(float(payload.get("min_mean_corr", 0.70))),
        "--max-rmse-ratio",
        str(float(payload.get("max_rmse_ratio", 0.08))),
        "--max-error-ratio",
        str(float(payload.get("max_error_ratio", 0.60))),
    ]
    shell_command = (
        "set +u; "
        "source /Users/kanghyunmin/miniconda3/bin/activate ros2_h311; "
        f"source {shlex.quote(str(overlay_setup))}; "
        f"cd {shlex.quote(str(PROJECT_ROOT))}; "
        + " ".join(shlex.quote(part) for part in command)
    )
    stdout = _run_command(["/bin/bash", "-lc", shell_command], timeout=180)
    result = _loads_json_from_stdout(stdout)

    if bool(payload.get("publish_live", False)):
        live_dir = PROJECT_ROOT / "outputs" / "live"
        live_dir.mkdir(parents=True, exist_ok=True)
        live_odometry = live_dir / "latest_odometry.csv"
        _atomic_copy(output_csv, live_odometry)
        publisher = restart_rviz_publisher(
            {
                "rate": float(payload.get("rate", 60.0)),
                "csv": str(live_odometry),
                "coordinate_frame": str(payload.get("coordinate_frame", "ros")),
            }
        )
        result["live_odometry_csv"] = str(live_odometry)
        result["publisher"] = publisher
    else:
        result["live_odometry_csv"] = ""
        result["publisher"] = None
    result["reference_publisher"] = restart_sim_reference_publisher(
        {
            "csv": str(dvl_reference_csv),
            "rate": float(payload.get("reference_rate", 1.0)),
            "line_width": float(payload.get("reference_line_width", 0.08)),
            "coordinate_frame": "flip-y",
        }
    )
    return result

def fit_to_sim_rc_replay(payload: dict[str, Any]) -> dict[str, Any]:
    input_csv = _require_project_file(payload.get("input_csv"))
    output_csv = _project_output_path(payload.get("output_csv", "outputs/live/latest_odometry_uuv_sim_fit.csv"))
    reference_csv = _project_output_path(payload.get("reference_csv", "outputs/evaluation/uuv_sim_rc_replay_reference.csv"))
    oracle_csv = PROJECT_ROOT / "outputs" / "live" / "latest_odometry_uuv_sim_oracle.csv"
    summary_json = PROJECT_ROOT / "outputs" / "evaluation" / "uuv_sim_rc_replay_fit_summary.json"

    command = [
        "python",
        str(PROJECT_ROOT / "scripts" / "sim_rc_replay_reference_pipeline.py"),
        "--input-csv",
        str(input_csv),
        "--output-csv",
        str(output_csv),
        "--oracle-csv",
        str(oracle_csv),
        "--reference-csv",
        str(reference_csv),
        "--summary-json",
        str(summary_json),
        "--reference-topic",
        str(payload.get("reference_topic", "/mujoco/ground_truth/pose")),
        "--time-map",
        str(payload.get("time_map", "normalized")),
        f"--time-offset-grid={payload.get('time_offset_grid', '0')}",
        "--offset-target",
        str(payload.get("offset_target", "reference")),
        "--metric-axes",
        str(payload.get("metric_axes", "xy")),
        "--selection-mode",
        str(payload.get("selection_mode", "max-corr")),
        "--output-z-mode",
        str(payload.get("output_z_mode", "zero")),
        "--max-step-grid",
        str(payload.get("max_step_grid", "0.03,0.05,0.08,0.10,0.12,0.15,0.20,0.30,0.40,0.50,0.75,1.0,1.5,2.0,1000.0")),
        "--fit-modes",
        str(payload.get("fit_modes", "rigid,sim3,affine")),
        "--min-mean-corr",
        str(float(payload.get("min_mean_corr", 0.70))),
        "--max-rmse-ratio",
        str(float(payload.get("max_rmse_ratio", 0.15))),
        "--max-error-ratio",
        str(float(payload.get("max_error_ratio", 0.80))),
    ]
    sim_bag_value = str(payload.get("sim_bag", "")).strip()
    if sim_bag_value and sim_bag_value.lower() != "latest":
        sim_bag = Path(sim_bag_value).expanduser().resolve()
        if not sim_bag.exists() or not sim_bag.is_file():
            raise ValueError(f"MuJoCo sim bag does not exist: {sim_bag}")
        command.extend(["--sim-bag", str(sim_bag)])

    shell_command = (
        "set +u; "
        "source /Users/kanghyunmin/miniconda3/bin/activate ros2_h311; "
        f"cd {shlex.quote(str(PROJECT_ROOT))}; "
        + " ".join(shlex.quote(part) for part in command)
    )
    stdout = _run_command(["/bin/bash", "-lc", shell_command], timeout=240)
    result = _loads_json_from_stdout(stdout)

    live_dir = PROJECT_ROOT / "outputs" / "live"
    live_dir.mkdir(parents=True, exist_ok=True)
    live_odometry = live_dir / "latest_odometry.csv"
    _atomic_copy(output_csv, live_odometry)
    publisher = restart_rviz_publisher(
        {
            "rate": float(payload.get("rate", 60.0)),
            "csv": str(live_odometry),
        }
    )
    result["live_odometry_csv"] = str(live_odometry)
    result["publisher"] = publisher
    return result


def publish_sim_rc_reference(payload: dict[str, Any]) -> dict[str, Any]:
    input_csv = _require_project_file(payload.get("input_csv"))
    reference_csv = _project_output_path(
        payload.get("reference_csv", "outputs/evaluation/uuv_sim_rc_replay_raw_xy_reference.csv")
    )
    output_csv = PROJECT_ROOT / "outputs" / "live" / "latest_odometry_uuv_sim_reference_red.csv"
    oracle_csv = PROJECT_ROOT / "outputs" / "live" / "latest_odometry_uuv_sim_reference_oracle.csv"
    summary_json = PROJECT_ROOT / "outputs" / "evaluation" / "uuv_sim_rc_replay_reference_publish_summary.json"
    time_offset = float(payload.get("time_offset_sec", 0.0))

    command = [
        "python",
        str(PROJECT_ROOT / "scripts" / "sim_rc_replay_reference_pipeline.py"),
        "--input-csv",
        str(input_csv),
        "--output-csv",
        str(output_csv),
        "--oracle-csv",
        str(oracle_csv),
        "--reference-csv",
        str(reference_csv),
        "--summary-json",
        str(summary_json),
        "--reference-topic",
        str(payload.get("reference_topic", "/mavros/local_position/odom")),
        "--time-map",
        str(payload.get("time_map", "normalized")),
        f"--time-offset-grid={time_offset:g}",
        "--metric-axes",
        str(payload.get("metric_axes", "xy")),
        "--selection-mode",
        "max-corr",
        "--output-z-mode",
        str(payload.get("output_z_mode", "zero")),
        "--max-step-grid",
        "1000.0",
        "--fit-modes",
        "oracle",
        "--min-mean-corr",
        "1.0",
        "--max-rmse-ratio",
        "0.000001",
        "--max-error-ratio",
        "0.000001",
    ]
    sim_bag_value = str(payload.get("sim_bag", "")).strip()
    if sim_bag_value and sim_bag_value.lower() != "latest":
        sim_bag = Path(sim_bag_value).expanduser().resolve()
        if not sim_bag.exists() or not sim_bag.is_file():
            raise ValueError(f"MuJoCo sim bag does not exist: {sim_bag}")
        command.extend(["--sim-bag", str(sim_bag)])

    shell_command = (
        "set +u; "
        "source /Users/kanghyunmin/miniconda3/bin/activate ros2_h311; "
        f"cd {shlex.quote(str(PROJECT_ROOT))}; "
        + " ".join(shlex.quote(part) for part in command)
    )
    stdout = _run_command(["/bin/bash", "-lc", shell_command], timeout=240)
    result = _loads_json_from_stdout(stdout)
    publisher = restart_sim_reference_publisher(
        {
            "csv": str(reference_csv),
            "rate": float(payload.get("rate", 1.0)),
            "line_width": float(payload.get("line_width", 0.08)),
        }
    )
    result["publisher"] = publisher
    return result


def restart_sim_reference_publisher(payload: dict[str, Any]) -> dict[str, Any]:
    rate = float(payload.get("rate", 1.0))
    if rate <= 0:
        raise ValueError("Sim reference publish rate must be positive.")
    csv_path = _require_project_file(str(payload.get("csv", "outputs/evaluation/uuv_sim_rc_replay_raw_xy_reference.csv")))
    line_width = float(payload.get("line_width", 0.08))
    coordinate_frame = str(payload.get("coordinate_frame", "ros"))
    if coordinate_frame not in {"ros", "opencv", "raw", "flip-y"}:
        raise ValueError(f"Unsupported static reference coordinate frame: {coordinate_frame}")

    for pid in _static_path_publisher_pids():
        try:
            os.kill(pid, 15)
        except ProcessLookupError:
            pass
    time.sleep(0.5)
    for pid in _static_path_publisher_pids():
        try:
            os.kill(pid, 9)
        except ProcessLookupError:
            pass

    log_path = Path("/tmp/underwater_sim_odom_reference.log")
    log_file = log_path.open("w", encoding="utf-8")
    path_topic = "/sim_odom_reference_path"
    marker_topic = "/sim_odom_reference_marker"
    command = (
        f"cd {shlex.quote(str(PROJECT_ROOT))} && "
        "set +u && "
        "source /Users/kanghyunmin/miniconda3/bin/activate ros2_h311 && "
        "export ROS_LOCALHOST_ONLY=1 && "
        "set -u && "
        f"exec python scripts/ros2_publish_static_path.py "
        f"--csv {shlex.quote(str(csv_path))} "
        f"--path-topic {shlex.quote(path_topic)} "
        f"--marker-topic {shlex.quote(marker_topic)} "
        f"--line-width {line_width} "
        f"--coordinate-frame {shlex.quote(coordinate_frame)} "
        f"--rate {rate}"
    )
    process = subprocess.Popen(
        ["/bin/bash", "-lc", command],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        cwd=PROJECT_ROOT,
        close_fds=True,
    )
    log_file.close()
    rows = _count_csv_rows(csv_path)
    return {
        "pid": process.pid,
        "csv": str(csv_path),
        "rate": rate,
        "rows": rows,
        "path_topic": path_topic,
        "marker_topic": marker_topic,
        "coordinate_frame": coordinate_frame,
        "log": str(log_path),
    }


def run_ros2_stereo_imu_vio(payload: dict[str, Any]) -> dict[str, Any]:
    bag_path = Path(str(payload["bag_path"])).expanduser().resolve()
    if not bag_path.exists() or not bag_path.is_file():
        raise ValueError(f"ROSBag db3 file does not exist: {bag_path}")

    timestamp = time.strftime("run_%Y%m%d_%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}"
    output_root = PROJECT_ROOT / "outputs" / "ros2_stereo_imu_vio" / "gui_runs" / timestamp
    dataset_dir = output_root / "stereo_imu_rosbag"
    odometry_csv = output_root / "ros2_stereo_imu_odometry.csv"
    summary_json = output_root / "ros2_stereo_imu_summary.json"
    debug_csv = output_root / "ros2_stereo_imu_debug.csv"

    export_command = (
        "set +u; "
        "source /Users/kanghyunmin/miniconda3/bin/activate ros2_h311; "
        f"python {shlex.quote(str(PROJECT_ROOT / 'scripts' / 'export_vins_stereo_imu_rosbag.py'))} "
        f"--bag {shlex.quote(str(bag_path))} "
        f"--output-dir {shlex.quote(str(dataset_dir))} "
        f"--max-frames {int(payload.get('max_frames', 316))} "
        f"--start-frame {int(payload.get('start_frame', 0))} "
        f"--stride {int(payload.get('stride', 1))} "
        f"--sync-tolerance-ms {float(payload.get('sync_tolerance_ms', 3.0))} "
        f"--imu-source {shlex.quote(str(payload.get('imu_source', 'realsense')))} "
        "--force"
    )
    export_stdout = _run_command(["/bin/bash", "-lc", export_command], timeout=300)
    export_metadata = json.loads(export_stdout)

    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "ros2_stereo_imu_vio.py"),
        "--dataset-dir",
        str(dataset_dir),
        "--output-csv",
        str(odometry_csv),
        "--summary-json",
        str(summary_json),
        "--debug-csv",
        str(debug_csv),
        "--feature-detector",
        str(payload.get("feature_detector", "orb")),
        "--depth-scale",
        str(float(payload.get("depth_scale", 0.85))),
        "--max-features",
        str(int(payload.get("max_features", 480))),
        "--min-features",
        str(int(payload.get("min_features", 120))),
        "--cell-max-features",
        str(int(payload.get("cell_max_features", 28))),
        "--min-distance",
        str(int(payload.get("min_distance", 16))),
        "--orb-fast-threshold",
        str(int(payload.get("orb_fast_threshold", 8))),
        "--fast-threshold",
        str(int(payload.get("fast_threshold", payload.get("orb_fast_threshold", 8)))),
        "--pnp-threshold",
        str(float(payload.get("pnp_threshold", 2.5))),
        "--min-pnp-inliers",
        str(int(payload.get("min_pnp_inliers", 18))),
        "--temporal-ransac",
        str(payload.get("temporal_ransac", "fundamental")),
        "--temporal-ransac-threshold",
        str(float(payload.get("temporal_ransac_threshold", 0.6))),
        "--min-temporal-inliers",
        str(int(payload.get("min_temporal_inliers", 18))),
        "--klt-max-flow-px",
        str(float(payload.get("klt_max_flow_px", 0.0))),
        "--klt-patch-ncc-threshold",
        str(float(payload.get("klt_patch_ncc_threshold", 0.0))),
        "--klt-patch-size",
        str(int(payload.get("klt_patch_size", 15))),
        "--klt-max-accel-px",
        str(float(payload.get("klt_max_accel_px", 25.0))),
        "--klt-min-eigenvalue",
        str(float(payload.get("klt_min_eigenvalue", 0.0))),
        "--min-track-age-for-pnp",
        str(int(payload.get("min_track_age_for_pnp", 1))),
        "--max-step-m",
        str(float(payload.get("max_step_m", 0.22))),
        "--max-speed-mps",
        str(float(payload.get("max_speed_mps", 1.0))),
        "--max-rotation-rad",
        str(float(payload.get("max_rotation_rad", 0.45))),
        "--imu-rotation-gate-rad",
        str(float(payload.get("imu_rotation_gate_rad", 0.75))),
        "--translation-smoothing",
        str(float(payload.get("translation_smoothing", 0.12))),
    ]
    if payload.get("reject_repeated_patches", False):
        command.append("--reject-repeated-patches")
    if payload.get("klt_local_flow_check", False):
        command.append("--klt-local-flow-check")
    if payload.get("klt_use_prediction", False):
        command.append("--klt-use-prediction")
    if payload.get("klt_velocity_check", False):
        command.append("--klt-velocity-check")
    stdout = _run_command(command, timeout=600)
    result = _loads_json_from_stdout(stdout)
    stats = _trajectory_sanity_stats(odometry_csv)
    valid, reason = _is_sane_vio_result(stats, export_metadata)

    live_dir = PROJECT_ROOT / "outputs" / "live"
    live_dir.mkdir(parents=True, exist_ok=True)
    live_odometry = live_dir / "latest_odometry.csv"
    live_pure_vins = live_dir / "latest_odometry_vins_pure_stereo_imu.csv"
    if valid:
        _atomic_copy(odometry_csv, live_odometry)
        _atomic_copy(odometry_csv, live_pure_vins)
        publisher = restart_rviz_publisher(
            {
                "rate": float(payload.get("rate", 10.0)),
                "csv": str(live_odometry),
                "coordinate_frame": "ros",
            }
        )
        overlay_publisher = restart_vins_dvl_overlay_publisher(
            {
                "vins_csv": str(live_pure_vins),
                "dvl_csv": "outputs/evaluation/dvl_xyz_30s_reference.csv",
            }
        )
    else:
        publisher = None
        overlay_publisher = None

    result["requested_max_frames"] = int(payload.get("max_frames", 316))
    result["dataset_dir"] = str(dataset_dir)
    result["output_dir"] = str(output_root)
    result["odometry_csv"] = str(odometry_csv)
    result["output_csv"] = str(odometry_csv)
    result["summary_json"] = str(summary_json)
    result["debug_csv"] = str(debug_csv)
    result["metadata"] = result.get("metadata", {})
    result["metadata"]["requested_max_frames"] = int(payload.get("max_frames", 316))
    result["trajectory_stats"] = stats
    result["trajectory_valid"] = valid
    result["trajectory_validation"] = reason
    result["live_odometry_csv"] = str(live_odometry) if valid else ""
    result["live_pure_vins_csv"] = str(live_pure_vins) if valid else ""
    result["publisher"] = publisher
    result["overlay_publisher"] = overlay_publisher
    result["export_metadata"] = export_metadata
    return result


def run_stereo_imu_vins(payload: dict[str, Any]) -> dict[str, Any]:
    bag_path = Path(str(payload["bag_path"])).expanduser().resolve()
    if not bag_path.exists() or not bag_path.is_file():
        raise ValueError(f"ROSBag db3 file does not exist: {bag_path}")

    if not _docker_available():
        raise RuntimeError(
            "Docker daemon is not running. Open Docker Desktop first, wait until it says Docker is running, then retry."
        )
    if not _docker_image_exists():
        build_vins_docker()

    timestamp = time.strftime("run_%Y%m%d_%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}"
    output_root = PROJECT_ROOT / "outputs" / "vins_fusion" / "gui_runs" / timestamp
    dataset_dir = output_root / "underwater_stereo_imu_rosbag"
    config_dir = output_root / "config"
    output_dir = output_root / "vins_output"
    odometry_csv = output_root / "vins_stereo_imu_odometry.csv"

    export_command = (
        "set +u; "
        "source /Users/kanghyunmin/miniconda3/bin/activate ros2_h311; "
        f"python {shlex.quote(str(PROJECT_ROOT / 'scripts' / 'export_vins_stereo_imu_rosbag.py'))} "
        f"--bag {shlex.quote(str(bag_path))} "
        f"--output-dir {shlex.quote(str(dataset_dir))} "
        f"--max-frames {int(payload.get('max_frames', 150))} "
        f"--start-frame {int(payload.get('start_frame', 0))} "
        f"--stride {int(payload.get('stride', 1))} "
        f"--sync-tolerance-ms {float(payload.get('sync_tolerance_ms', 3.0))} "
        f"--imu-source {shlex.quote(str(payload.get('imu_source', 'realsense')))} "
        "--force"
    )
    export_stdout = _run_command(["/bin/bash", "-lc", export_command], timeout=300)
    metadata = json.loads(export_stdout)
    requested_max_frames = int(payload.get("max_frames", 150))

    config_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "vins_make_stereo_imu_config.py"),
        "--metadata",
        str(dataset_dir / "metadata.json"),
        "--config-dir",
        str(config_dir),
        "--output-dir",
        str(output_dir),
        "--estimate-extrinsic",
        str(int(payload.get("estimate_extrinsic", 1))),
        "--estimate-td",
        str(int(payload.get("estimate_td", 1))),
        "--max-cnt",
        str(int(payload["max_cnt"])),
        "--min-dist",
        str(int(payload["min_dist"])),
        "--freq",
        str(int(payload.get("freq", 10))),
        "--f-threshold",
        str(float(payload["f_threshold"])),
        "--max-solver-time",
        str(float(payload["max_solver_time"])),
        "--max-num-iterations",
        str(int(payload["max_num_iterations"])),
        "--keyframe-parallax",
        str(float(payload["keyframe_parallax"])),
        "--acc-n",
        str(float(payload.get("acc_n", 0.1))),
        "--gyr-n",
        str(float(payload.get("gyr_n", 0.01))),
        "--acc-w",
        str(float(payload.get("acc_w", 0.001))),
        "--gyr-w",
        str(float(payload.get("gyr_w", 0.0001))),
    ]
    config_file = Path(_run_command(config_cmd, timeout=60).strip()).resolve()

    env: dict[str, str] = {}
    env["VINS_REPLAY_REALTIME_FACTOR"] = str(float(payload.get("replay_realtime_factor", 12.0)))
    docker_cmd = [
        str(PROJECT_ROOT / "scripts" / "vins_docker_run_stereo_imu_rosbag.sh"),
        str(dataset_dir),
        str(config_file),
        str(output_dir),
    ]
    vins_log = _tail_lines(_run_command(docker_cmd, timeout=1200, env=env), 100)

    convert_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "convert_vins_vio_to_csv.py"),
        "--vio-txt",
        str(output_dir / "vio.csv"),
        "--times-txt",
        str(dataset_dir / "times.txt"),
        "--output-csv",
        str(odometry_csv),
        "--scale-mode",
        "vins_fusion_stereo_imu",
    ]
    _run_command(convert_cmd, timeout=60)
    stats = _trajectory_sanity_stats(odometry_csv)
    valid, reason = _is_sane_vio_result(stats, metadata)
    pose_count = max(0, sum(1 for _ in odometry_csv.open(encoding="utf-8")) - 1)
    if not valid:
        return {
            "mode": "stereo_imu_rosbag",
            "requested_max_frames": requested_max_frames,
            "metadata": metadata,
            "dataset_dir": str(dataset_dir),
            "config_file": str(config_file),
            "output_dir": str(output_dir),
            "odometry_csv": str(odometry_csv),
            "live_odometry_csv": "",
            "pose_count": pose_count,
            "trajectory_stats": stats,
            "trajectory_valid": False,
            "trajectory_validation": reason,
            "vins_log": (vins_log + "\n\nRejected trajectory: " + reason).strip(),
        }

    live_dir = PROJECT_ROOT / "outputs" / "live"
    live_dir.mkdir(parents=True, exist_ok=True)
    live_odometry = live_dir / "latest_odometry.csv"
    _atomic_copy(odometry_csv, live_odometry)

    return {
        "mode": "stereo_imu_rosbag",
        "requested_max_frames": requested_max_frames,
        "metadata": metadata,
        "dataset_dir": str(dataset_dir),
        "config_file": str(config_file),
        "output_dir": str(output_dir),
        "odometry_csv": str(odometry_csv),
        "live_odometry_csv": str(live_odometry),
        "pose_count": pose_count,
        "trajectory_stats": stats,
        "trajectory_valid": True,
        "trajectory_validation": reason,
        "vins_log": vins_log,
    }


def _atomic_copy(source: Path, destination: Path) -> None:
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copyfile(source, tmp)
    tmp.replace(destination)


def _trajectory_sanity_stats(csv_path: Path) -> dict[str, float]:
    rows: list[tuple[float, float, float, float]] = []
    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            rows.append((float(row["timestamp_sec"]), float(row["x"]), float(row["y"]), float(row["z"])))
    if len(rows) < 2:
        return {
            "rows": float(len(rows)),
            "duration_sec": 0.0,
            "path_length_m": 0.0,
            "max_step_m": 0.0,
            "max_speed_mps": 0.0,
            "span_x_m": 0.0,
            "span_y_m": 0.0,
            "span_z_m": 0.0,
        }
    path_length = 0.0
    max_step = 0.0
    max_speed = 0.0
    for prev, cur in zip(rows, rows[1:]):
        dt = max(cur[0] - prev[0], 1e-6)
        step = ((cur[1] - prev[1]) ** 2 + (cur[2] - prev[2]) ** 2 + (cur[3] - prev[3]) ** 2) ** 0.5
        path_length += step
        max_step = max(max_step, step)
        max_speed = max(max_speed, step / dt)
    xs = [row[1] for row in rows]
    ys = [row[2] for row in rows]
    zs = [row[3] for row in rows]
    return {
        "rows": float(len(rows)),
        "duration_sec": rows[-1][0] - rows[0][0],
        "path_length_m": path_length,
        "max_step_m": max_step,
        "max_speed_mps": max_speed,
        "span_x_m": max(xs) - min(xs),
        "span_y_m": max(ys) - min(ys),
        "span_z_m": max(zs) - min(zs),
    }


def _is_sane_vio_result(stats: dict[str, float], metadata: dict[str, Any]) -> tuple[bool, str]:
    duration = max(float(stats["duration_sec"]), 1e-6)
    mean_speed = float(stats["path_length_m"]) / duration
    max_span = max(float(stats["span_x_m"]), float(stats["span_y_m"]), float(stats["span_z_m"]))
    expected_frames = int(metadata.get("frames") or 0)
    if float(stats["rows"]) < 20:
        return False, f"too few VINS poses ({int(stats['rows'])})"
    if expected_frames >= 50 and float(stats["rows"]) < expected_frames * 0.9:
        return False, f"VINS only produced {int(stats['rows'])}/{expected_frames} poses"
    if metadata.get("format") == "vins_fusion_kitti_stereo":
        if float(stats["path_length_m"]) > 1000.0:
            return False, f"path length too large: {stats['path_length_m']:.2f} m"
        if max_span > 1000.0:
            return False, f"trajectory span too large: {max_span:.2f} m"
        return True, f"stereo-only path accepted: path_length={stats['path_length_m']:.2f} m"

    reference_length = _reference_path_length(metadata)
    if reference_length is not None:
        allowed_length = max(50.0, reference_length * 3.0)
        if float(stats["path_length_m"]) > allowed_length:
            return (
                False,
                f"path length inconsistent with /odometry/filtered: "
                f"vins={stats['path_length_m']:.2f} m, reference={reference_length:.2f} m",
            )
    if float(stats["max_step_m"]) > 5.0:
        return False, f"pose jump too large: max_step={stats['max_step_m']:.2f} m"
    if float(stats["max_speed_mps"]) > 5.0:
        return False, f"instant speed too high: max_speed={stats['max_speed_mps']:.2f} m/s"
    if mean_speed > 2.5:
        return False, f"mean path speed too high: {mean_speed:.2f} m/s"
    if max_span > 80.0:
        return False, f"trajectory span too large: {max_span:.2f} m"
    if metadata.get("last_imu_stamp_ns") and metadata.get("last_image_stamp_ns"):
        if int(metadata["last_imu_stamp_ns"]) + 20_000_000 < int(metadata["last_image_stamp_ns"]):
            return False, "image stream extends past usable IMU coverage"
    return True, f"sane trajectory: mean_speed={mean_speed:.2f} m/s, max_step={stats['max_step_m']:.2f} m"


def _reference_path_length(metadata: dict[str, Any]) -> float | None:
    start_ns = metadata.get("first_image_stamp_ns")
    end_ns = metadata.get("last_image_stamp_ns")
    reference_csv = PROJECT_ROOT / "outputs" / "evaluation" / "rosbag_odometry_filtered_reference.csv"
    if not start_ns or not end_ns or not reference_csv.exists():
        return None
    start = int(start_ns)
    end = int(end_ns)
    points: list[tuple[float, float, float]] = []
    with reference_csv.open(newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            timestamp_ns = int(row["bag_timestamp_ns"])
            if start <= timestamp_ns <= end:
                points.append((float(row["x"]), float(row["y"]), float(row["z"])))
    if len(points) < 2:
        return None
    total = 0.0
    for prev, cur in zip(points, points[1:]):
        total += ((cur[0] - prev[0]) ** 2 + (cur[1] - prev[1]) ** 2 + (cur[2] - prev[2]) ** 2) ** 0.5
    return total


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return float(text)


def _ros2_publisher_pids() -> list[int]:
    completed = subprocess.run(
        ["ps", "-ax", "-o", "pid=,command="],
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    pids: list[int] = []
    for line in completed.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        pid_text, _, command = stripped.partition(" ")
        if "ros2_publish_odometry.py" not in command:
            continue
        executable = command.split(maxsplit=1)[0].lower()
        if "python" not in Path(executable).name:
            continue
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        if pid != os.getpid():
            pids.append(pid)
    return pids


def _path_overlay_publisher_pids() -> list[int]:
    completed = subprocess.run(
        ["ps", "-ax", "-o", "pid=,command="],
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    pids: list[int] = []
    for line in completed.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        pid_text, _, command = stripped.partition(" ")
        if "ros2_publish_path_overlay.py" not in command:
            continue
        executable = command.split(maxsplit=1)[0].lower()
        if "python" not in Path(executable).name:
            continue
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        if pid != os.getpid():
            pids.append(pid)
    return pids


def _static_path_publisher_pids() -> list[int]:
    completed = subprocess.run(
        ["ps", "-ax", "-o", "pid=,command="],
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    pids: list[int] = []
    for line in completed.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        pid_text, _, command = stripped.partition(" ")
        if "ros2_publish_static_path.py" not in command:
            continue
        executable = command.split(maxsplit=1)[0].lower()
        if "python" not in Path(executable).name:
            continue
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        if pid != os.getpid():
            pids.append(pid)
    return pids


def _count_csv_rows(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as csv_file:
        return max(0, sum(1 for _ in csv_file) - 1)


def _run_command(command: list[str], timeout: int, env: dict[str, str] | None = None) -> str:
    command_env = None
    if env:
        command_env = {**dict(os.environ), **env}
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=command_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            + " ".join(command)
            + "\n\n"
            + _tail_lines(completed.stdout, 40)
        )
    return completed.stdout


def _loads_json_from_stdout(stdout: str) -> dict[str, Any]:
    start = stdout.find("{")
    end = stdout.rfind("}")
    if start < 0 or end < start:
        raise RuntimeError("Command did not return JSON:\n" + _tail_lines(stdout, 40))
    return json.loads(stdout[start : end + 1])


def _tail_lines(text: str, count: int) -> str:
    lines = text.splitlines()
    return "\n".join(lines[-count:])


def _require_project_file(value: Any) -> Path:
    path = Path(str(value)).resolve()
    if not _is_under_project(path) or not path.exists() or not path.is_file():
        raise ValueError(f"File is not available in this project: {value}")
    return path


def _project_output_path(value: Any) -> Path:
    path = Path(str(value)).resolve()
    if not _is_under_project(path):
        raise ValueError(f"Output path must stay inside this project: {value}")
    return path


def _docker_available() -> bool:
    return subprocess.run(
        ["docker", "info"],
        cwd=PROJECT_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def _docker_image_exists() -> bool:
    return subprocess.run(
        ["docker", "image", "ls", "-q", "underwater-vins-fusion:kinetic"],
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
        text=True,
    ).stdout.strip() != ""


def _feature_benchmark() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    output_root = PROJECT_ROOT / "outputs"
    if not output_root.exists():
        return {
            "recommended_method": RECOMMENDED_FEATURE_METHOD,
            "recommended": None,
            "best_by_method": [],
            "sample_count": 0,
        }

    for summary_path in output_root.rglob("*summary.json"):
        if "vins_fusion" in summary_path.parts:
            continue
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        method = data.get("feature_method")
        frames_processed = _safe_int(data.get("frames_processed"))
        if not method or frames_processed < 50:
            continue

        pose_success_ratio = _safe_float(data.get("pose_success_ratio"))
        mean_inlier_ratio = _safe_float(data.get("mean_geometry_inlier_ratio"))
        mean_inliers = _safe_float(data.get("mean_geometry_inliers"))
        mean_tracks = _safe_float(data.get("mean_active_tracks"))
        if pose_success_ratio is None or mean_inlier_ratio is None:
            continue

        input_path = Path(str(data.get("input", "")))
        rows.append(
            {
                "method": str(method),
                "frames_processed": frames_processed,
                "pose_success_ratio": pose_success_ratio,
                "mean_geometry_inlier_ratio": mean_inlier_ratio,
                "mean_geometry_inliers": mean_inliers or 0.0,
                "mean_active_tracks": mean_tracks or 0.0,
                "input_name": input_path.name,
                "summary_name": summary_path.name,
                "summary_path": str(summary_path),
            }
        )

    best_by_method: list[dict[str, Any]] = []
    for method in sorted({row["method"] for row in rows}, key=lambda name: FEATURE_METHOD_ORDER.get(name, 99)):
        method_rows = [row for row in rows if row["method"] == method]
        best_by_method.append(max(method_rows, key=_benchmark_score))

    recommended_rows = [row for row in rows if row["method"] == RECOMMENDED_FEATURE_METHOD]
    recommended = max(recommended_rows, key=_benchmark_score) if recommended_rows else None
    return {
        "recommended_method": RECOMMENDED_FEATURE_METHOD,
        "recommended": recommended,
        "best_by_method": best_by_method,
        "sample_count": len(rows),
    }


def _trajectory_accuracy_benchmark() -> dict[str, Any]:
    path = PROJECT_ROOT / "outputs" / "evaluation" / "trajectory_accuracy_all_runs.csv"
    if not path.exists():
        return {"source": str(path), "rows": []}

    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            duration = _safe_float(row.get("duration_sec")) or 0.0
            if duration < 10.0:
                continue
            parsed = dict(row)
            for key in (
                "frames",
                "duration_sec",
                "ref_path_length",
                "est_path_length",
                "raw_rmse",
                "se3_rmse",
                "sim3_rmse",
                "sim3_median",
                "sim3_scale",
            ):
                parsed[key] = _safe_float(parsed.get(key)) or 0.0
            rows.append(parsed)

    rows.sort(key=lambda item: (float(item["sim3_rmse"]), -float(item["duration_sec"])))
    return {"source": str(path), "rows": rows}


def _benchmark_score(row: dict[str, Any]) -> tuple[float, float, int]:
    return (
        float(row["pose_success_ratio"]),
        float(row["mean_geometry_inlier_ratio"]),
        int(row["frames_processed"]),
    )


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _list_videos() -> list[dict[str, str]]:
    videos = sorted((PROJECT_ROOT / "video_src").glob("*.mp4"))
    return [{"name": path.name, "path": str(path)} for path in videos]


def _is_under_project(path: Path) -> bool:
    try:
        path.relative_to(PROJECT_ROOT)
        return True
    except ValueError:
        return False


def _content_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".html":
        return "text/html; charset=utf-8"
    if suffix == ".json":
        return "application/json; charset=utf-8"
    if suffix == ".csv":
        return "text/csv; charset=utf-8"
    if suffix == ".jpg" or suffix == ".jpeg":
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix == ".mp4":
        return "video/mp4"
    return "application/octet-stream"


def _parse_range_header(range_header: str, file_size: int) -> tuple[int, int]:
    if not range_header.startswith("bytes="):
        return 0, file_size - 1
    value = range_header.removeprefix("bytes=").split(",", 1)[0].strip()
    start_text, _, end_text = value.partition("-")
    if start_text == "":
        suffix_length = int(end_text)
        return max(0, file_size - suffix_length), file_size - 1
    start = int(start_text)
    end = int(end_text) if end_text else file_size - 1
    start = max(0, min(start, file_size - 1))
    end = max(start, min(end, file_size - 1))
    return start, end


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Browser GUI for VINS-Fusion stereo MP4 testing.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--max-port-tries",
        type=int,
        default=20,
        help="Try this many sequential ports if the requested port is busy.",
    )
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()

    server, port = _bind_server(args.host, args.port, args.max_port_tries)
    url = f"http://{args.host}:{port}"
    print(f"Open {url}", flush=True)
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping web GUI")
    finally:
        server.server_close()


def _bind_server(host: str, port: int, max_port_tries: int) -> tuple[ThreadingHTTPServer, int]:
    last_error = None
    for candidate in range(port, port + max_port_tries):
        try:
            return ThreadingHTTPServer((host, candidate), WebGuiHandler), candidate
        except OSError as exc:
            last_error = exc
            if exc.errno not in {48, 98}:
                raise
            print(f"Port {candidate} is busy, trying {candidate + 1}", flush=True)
    raise OSError(f"Could not bind web GUI after {max_port_tries} tries") from last_error


if __name__ == "__main__":
    main()
