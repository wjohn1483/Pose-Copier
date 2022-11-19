/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import '@tensorflow/tfjs-backend-webgl';
import * as mpPose from '@mediapipe/pose';

import * as tfjsWasm from '@tensorflow/tfjs-backend-wasm';

tfjsWasm.setWasmPaths(
    `https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@${
        tfjsWasm.version_wasm}/dist/`);

import * as posedetection from '@tensorflow-models/pose-detection';

import {Camera} from './camera';
import {setupDatGui} from './option_panel';
import {STATE} from './params';
import {setupStats} from './stats_panel';
import {setBackendAndEnvFlags} from './util';
import {showImage} from './image';

let detector, camera, stats, distance_score_header;
let image_poses, camera_poses;
let startInferenceTime, numInferences = 0;
let inferenceTimeSum = 0, lastPanelUpdate = 0;
let rafId;
const consine_similarity = require('compute-cosine-similarity');

async function createDetector() {
  switch (STATE.model) {
    case posedetection.SupportedModels.PoseNet:
      return posedetection.createDetector(STATE.model, {
        quantBytes: 4,
        architecture: 'MobileNetV1',
        outputStride: 16,
        inputResolution: {width: 500, height: 500},
        multiplier: 0.75
      });
    case posedetection.SupportedModels.BlazePose:
      const runtime = STATE.backend.split('-')[0];
      if (runtime === 'mediapipe') {
        return posedetection.createDetector(STATE.model, {
          runtime,
          modelType: STATE.modelConfig.type,
          solutionPath: `https://cdn.jsdelivr.net/npm/@mediapipe/pose@${mpPose.VERSION}`
        });
      } else if (runtime === 'tfjs') {
        return posedetection.createDetector(
            STATE.model, {runtime, modelType: STATE.modelConfig.type});
      }
    case posedetection.SupportedModels.MoveNet:
      let modelType;
      if (STATE.modelConfig.type == 'lightning') {
        modelType = posedetection.movenet.modelType.SINGLEPOSE_LIGHTNING;
      } else if (STATE.modelConfig.type == 'thunder') {
        modelType = posedetection.movenet.modelType.SINGLEPOSE_THUNDER;
      } else if (STATE.modelConfig.type == 'multipose') {
        modelType = posedetection.movenet.modelType.MULTIPOSE_LIGHTNING;
      }
      const modelConfig = {modelType};

      if (STATE.modelConfig.customModel !== '') {
        modelConfig.modelUrl = STATE.modelConfig.customModel;
      }
      if (STATE.modelConfig.type === 'multipose') {
        modelConfig.enableTracking = STATE.modelConfig.enableTracking;
      }
      return posedetection.createDetector(STATE.model, modelConfig);
  }
}

async function checkGuiUpdate() {
  if (STATE.isTargetFPSChanged || STATE.isSizeOptionChanged) {
    camera = await Camera.setupCamera(STATE.camera);
    STATE.isTargetFPSChanged = false;
    STATE.isSizeOptionChanged = false;
  }

  if (STATE.isModelChanged || STATE.isFlagChanged || STATE.isBackendChanged) {
    STATE.isModelChanged = true;

    window.cancelAnimationFrame(rafId);

    if (detector != null) {
      detector.dispose();
    }

    if (STATE.isFlagChanged || STATE.isBackendChanged) {
      await setBackendAndEnvFlags(STATE.flags, STATE.backend);
    }

    try {
      detector = await createDetector(STATE.model);
    } catch (error) {
      detector = null;
      alert(error);
    }

    STATE.isFlagChanged = false;
    STATE.isBackendChanged = false;
    STATE.isModelChanged = false;
  }
}

function beginEstimatePosesStats() {
  startInferenceTime = (performance || Date).now();
}

function endEstimatePosesStats() {
  const endInferenceTime = (performance || Date).now();
  inferenceTimeSum += endInferenceTime - startInferenceTime;
  ++numInferences;

  const panelUpdateMilliseconds = 1000;
  if (endInferenceTime - lastPanelUpdate >= panelUpdateMilliseconds) {
    const averageInferenceTime = inferenceTimeSum / numInferences;
    inferenceTimeSum = 0;
    numInferences = 0;
    stats.customFpsPanel.update(
        1000.0 / averageInferenceTime, 120 /* maxValue */);
    lastPanelUpdate = endInferenceTime;
  }
}

async function predictImagePoses() {
  let poses = null;

  // Detector can be null if initialization failed (for example when loading
  // from a URL that does not exist).
  if (detector != null) {
    // Detectors can throw errors, for example when using custom URLs that
    // contain a model that doesn't provide the expected output.
    try {
      poses = await detector.estimatePoses(
          photo_image,
          {maxPoses: STATE.modelConfig.maxPoses, flipHorizontal: false});
    } catch (error) {
      detector.dispose();
      detector = null;
      alert(error);
    }
  }
  return poses;
}

async function predictCameraPoses() {
  if (camera.video.readyState < 2) {
    await new Promise((resolve) => {
      camera.video.onloadeddata = () => {
        resolve(video);
      };
    });
  }

  let poses = null;

  // Detector can be null if initialization failed (for example when loading
  // from a URL that does not exist).
  if (detector != null) {
    // FPS only counts the time it takes to finish estimatePoses.
    beginEstimatePosesStats();

    // Detectors can throw errors, for example when using custom URLs that
    // contain a model that doesn't provide the expected output.
    try {
      poses = await detector.estimatePoses(
          camera.video,
          {maxPoses: STATE.modelConfig.maxPoses, flipHorizontal: false});
    } catch (error) {
      detector.dispose();
      detector = null;
      alert(error);
    }

    endEstimatePosesStats();
  }
  return poses;
}

async function updateWindow(poses, distance) {
  distance_score_header.innerText = "Distance = " + distance;
  camera.drawCtx();
  // The null check makes sure the UI is not in the middle of changing to a
  // different model. If during model change, the result is from an old model,
  // which shouldn't be rendered.
  if (poses && poses.length > 0 && !STATE.isModelChanged) {
    camera.drawResults(poses);
  }
}

async function convertPoseToVector(pose) {
  var vector = [];
  var confidence_score = [];
  var norm = 0;
  var sum_of_confidence_score = 0;
  var keypoints = pose[0]["keypoints"];
  var array_length = keypoints.length;
  for (var i = 0; i < array_length; i++) {
    vector.push(keypoints[i]["x"]);
    vector.push(keypoints[i]["y"]);
    norm += keypoints[i]["x"]*keypoints[i]["x"];
    norm += keypoints[i]["y"]*keypoints[i]["y"];
    confidence_score.push(keypoints[i]["score"]);
    sum_of_confidence_score += keypoints[i]["score"];
  }
  norm = Math.sqrt(norm);
  for (var i = 0; i < vector.length; i++) {
    vector[i] /= norm;
  }
  return [vector, confidence_score, sum_of_confidence_score, confidence_score.length];
}

async function calculateConsineDistance(poseVector1, poseVector2) {
  let cosineSimilarity = consine_similarity(poseVector1, poseVector2);
  let distance = 2 * (1 - cosineSimilarity);
  return Math.sqrt(distance);
}

async function calculateWeightedDistance(poseVector1, poseVector2, num_of_body_part) {
  // poseVector1 and poseVector2 are 52-float vectors composed of:
  // Values 0-33: are x,y coordinates for 17 body parts in alphabetical order
  // Values 34-51: are confidence values for each of the 17 body parts in alphabetical order
  // Value 51: A sum of all the confidence values
  // Again the lower the number, the closer the distance

  if (!poseVector1) {
    console.log("calculate NULL");
    return 0;
  }
  let vector1PoseXY = poseVector1.slice(0, 2*num_of_body_part);
  let vector1Confidences = poseVector1.slice(2*num_of_body_part, 3*num_of_body_part);
  let vector1ConfidenceSum = poseVector1.slice(3*num_of_body_part, 3*num_of_body_part+1);

  let vector2PoseXY = poseVector2.slice(0, 2*num_of_body_part);

  // First summation
  let summation1 = 1 / vector1ConfidenceSum;

  // Second summation
  let summation2 = 0;
  for (let i = 0; i < vector1PoseXY.length; i++) {
    let tempConf = Math.floor(i / 2);
    let tempSum = vector1Confidences[tempConf] * Math.abs(vector1PoseXY[i] - vector2PoseXY[i]);
    summation2 = summation2 + tempSum;
  }

  return summation1 * summation2;
}

async function renderPrediction() {
  await checkGuiUpdate();

  if (STATE.isImageChanged) {
    image_poses = await predictImagePoses();
    STATE.isImageChanged = false;
  }
  if (!STATE.isModelChanged) {
    camera_poses = await predictCameraPoses();
  }

  if (image_poses && camera_poses) {
    var image_result = await convertPoseToVector(image_poses);
    var camera_result = await convertPoseToVector(camera_poses);

    // // Weighed distance
    // var image_vector = image_result[0].concat(image_result[1]).concat(image_result[2]);
    // var camera_vector = camera_result[0].concat(camera_result[1]).concat(camera_result[2]);
    // var num_of_body_part = image_result[3];
    // var distance = await calculateWeightedDistance(image_vector, camera_vector, num_of_body_part);

    // Consine distance
    var image_vector = image_result[0];
    var camera_vector = camera_result[0];
    var distance = await calculateConsineDistance(image_vector, camera_vector);
  } else {
    var distance = "NaN";
  }

  await updateWindow(camera_poses, distance);

  rafId = requestAnimationFrame(renderPrediction);
};

async function app() {
  // Gui content will change depending on which model is in the query string.
  const urlParams = new URLSearchParams(window.location.search);
  if (!urlParams.has('model')) {
    alert('Cannot find model in the query string.');
    return;
  }

  await setupDatGui(urlParams);

  stats = setupStats();

  camera = await Camera.setupCamera(STATE.camera);

  await setBackendAndEnvFlags(STATE.flags, STATE.backend);

  detector = await createDetector();

  renderPrediction();

  const photo_src = document.getElementById("photo_src");
  const photo_image = document.getElementById("photo_image");
  distance_score_header = document.getElementById("distance_score_header");

  showImage(photo_src, photo_image);
};

app();
