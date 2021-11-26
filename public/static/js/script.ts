import '@tensorflow/tfjs-backend-webgl';
import '@tensorflow/tfjs-backend-cpu';

import * as faceLandmarksDetection from '@tensorflow-models/face-landmarks-detection';
import * as tfjsWasm from '@tensorflow/tfjs-backend-wasm';
import * as tf from '@tensorflow/tfjs-core';
import Stats from 'stats.js';
import dat from 'dat.gui'

import { AnnotatedPrediction, MediaPipeFaceMesh } from '@tensorflow-models/face-landmarks-detection/dist/mediapipe-facemesh';
import { Coord2D, Coords3D } from '@tensorflow-models/face-landmarks-detection/dist/mediapipe-facemesh/util';

tfjsWasm.setWasmPaths(
    `https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@${
        tfjsWasm.version_wasm}/dist/`);

const NUM_KEYPOINTS = 468;
const NUM_IRIS_KEYPOINTS = 5;
const GREEN = '#32EEDB';
const RED = '#FF2C35';
const BLUE = '#157AB3';
let stopRendering = false;

function isMobile() {
  const isAndroid = /Android/i.test(navigator.userAgent);
  const isiOS = /iPhone|iPad|iPod/i.test(navigator.userAgent);
  return isAndroid || isiOS;
}

function distance(a: number[], b: number[]) {
  return Math.sqrt(Math.pow(a[0] - b[0], 2) + Math.pow(a[1] - b[1], 2));
}

function drawPath(ctx: CanvasRenderingContext2D, points: number[][], closePath: boolean) {
  const region = new Path2D();
  region.moveTo(points[0][0], points[0][1]);
  for (let i = 1; i < points.length; i++) {
    const point = points[i];
    region.lineTo(point[0], point[1]);
  }

  if (closePath) {
    region.closePath();
  }
  ctx.stroke(region);
}

let model: MediaPipeFaceMesh,
    ctx: CanvasRenderingContext2D,
    videoWidth: number,
    videoHeight: number,
    video: HTMLVideoElement,
    canvas: HTMLCanvasElement,
    rafID: number;

const VIDEO_SIZE = 500;
const mobile = isMobile();

const stats = new Stats();
const state = {
  backend: 'webgl',
  maxFaces: 1,
  validation: {
    blinkCount: 3
  },
  currentStatus: {
    blinkCount: 0,
    lastEyeStateOpen: true
  }
};

async function setupCamera() {
  video = document.getElementById('video') as HTMLVideoElement;

  const stream = await navigator.mediaDevices.getUserMedia({
    'audio': false,
    'video': {
      // 'deviceId': '26ac4c7879bc217604ddb8618960818bb9a7f5ed972a585f0d844c17b8f3c128',
      facingMode: 'user',
      // Only setting the video to a specified size in order to accommodate a
      // point cloud, so on mobile devices accept the default size.
      // width: mobile ? undefined : VIDEO_SIZE,
      // height: mobile ? undefined : VIDEO_SIZE
    },
  });
  video.srcObject = stream;

  // video.src = 'static/videos/VID_20211112_070051.mp4'
  return new Promise((resolve) => {
    video.onloadedmetadata = () => {
      resolve(video);
    };
  });
}

async function renderPrediction() {
  if (stopRendering) {
    return;
  }

  stats.begin();

  const predictions:AnnotatedPrediction[] = await model.estimateFaces({
    input: video,
    returnTensors: false,
    flipHorizontal: false,
    predictIrises: true
  });

  // @ts-ignore
  window.debug && console.log(predictions)

  ctx.drawImage(
      video, 0, 0, videoWidth, videoHeight, 0, 0, canvas.width, canvas.height);


  if (predictions.length > 0) {
    calculateLiveliness(predictions)
    predictions.forEach(prediction => {
      const topLeft = prediction.boundingBox.topLeft as Coord2D;
      const bottomRight = prediction.boundingBox.bottomRight as Coord2D;
      const keypoints:Coords3D = prediction.scaledMesh as Coords3D;

      // Bounding Box
      ctx.strokeStyle = GREEN;
      ctx.beginPath();
      ctx.rect(topLeft[0], topLeft[1], bottomRight[0], bottomRight[1]);
      ctx.stroke();

      ctx.strokeStyle = RED;
      // @ts-ignore
      let leftEyeLower = prediction.annotations.leftEyeLower0, leftEyeUpper = prediction.annotations.leftEyeUpper0;
      // @ts-ignore
      let rightEyeLower = prediction.annotations.rightEyeLower0, rightEyeUpper = prediction.annotations.rightEyeUpper0;

      let leftEyeLowerMiddle = leftEyeLower[Math.floor(leftEyeLower.length/2)]
      let leftEyeUpperMiddle = leftEyeUpper[Math.floor(leftEyeUpper.length/2)]

      let rightEyeLowerMiddle = rightEyeLower[Math.floor(rightEyeLower.length/2)]
      let rightEyeUpperMiddle = rightEyeUpper[Math.floor(rightEyeUpper.length/2)]

      let leftGap = distance(leftEyeLowerMiddle, leftEyeUpperMiddle);
      let rightGap = distance(rightEyeLowerMiddle, rightEyeUpperMiddle);
      // console.log('Left: ', leftGap > 5 ? 'open' : 'closed')
      // console.log('Right: ', rightGap > 5 ? 'open' : 'closed')

      drawPath(ctx, leftEyeUpper, false)
      drawPath(ctx, leftEyeLower, false)
      drawPath(ctx, rightEyeUpper, false)
      drawPath(ctx, rightEyeLower, false)


      if (keypoints.length > NUM_KEYPOINTS) {
        ctx.strokeStyle = RED;
        ctx.lineWidth = 1;

        const leftCenter = keypoints[NUM_KEYPOINTS];
        const leftDiameterY = distance(
            keypoints[NUM_KEYPOINTS + 4], keypoints[NUM_KEYPOINTS + 2]);
        const leftDiameterX = distance(
            keypoints[NUM_KEYPOINTS + 3], keypoints[NUM_KEYPOINTS + 1]);

        ctx.beginPath();
        ctx.ellipse(
            leftCenter[0], leftCenter[1], leftDiameterX / 2, leftDiameterY / 2,
            0, 0, 2 * Math.PI);
        ctx.stroke();

        if (keypoints.length > NUM_KEYPOINTS + NUM_IRIS_KEYPOINTS) {
          const rightCenter = keypoints[NUM_KEYPOINTS + NUM_IRIS_KEYPOINTS];
          const rightDiameterY = distance(
              keypoints[NUM_KEYPOINTS + NUM_IRIS_KEYPOINTS + 2],
              keypoints[NUM_KEYPOINTS + NUM_IRIS_KEYPOINTS + 4]);
          const rightDiameterX = distance(
              keypoints[NUM_KEYPOINTS + NUM_IRIS_KEYPOINTS + 3],
              keypoints[NUM_KEYPOINTS + NUM_IRIS_KEYPOINTS + 1]);

          ctx.beginPath();
          ctx.ellipse(
              rightCenter[0], rightCenter[1], rightDiameterX / 2,
              rightDiameterY / 2, 0, 0, 2 * Math.PI);
          ctx.stroke();
        }
      }

      if(true) {
        ctx.strokeStyle = BLUE;
        ctx.lineWidth = 1;

        const leftCenter = keypoints[NUM_KEYPOINTS];
        const leftDiameterY = distance(
            keypoints[NUM_KEYPOINTS + 4], keypoints[NUM_KEYPOINTS + 2]);
        const leftDiameterX = distance(
            keypoints[NUM_KEYPOINTS + 3], keypoints[NUM_KEYPOINTS + 1]);

        ctx.beginPath();
        ctx.ellipse(
            leftCenter[0], leftCenter[1], leftDiameterX / 2, leftDiameterY / 2,
            0, 0, 2 * Math.PI);
        ctx.stroke();

        if (keypoints.length > NUM_KEYPOINTS + NUM_IRIS_KEYPOINTS) {
          const rightCenter = keypoints[NUM_KEYPOINTS + NUM_IRIS_KEYPOINTS];
          const rightDiameterY = distance(
              keypoints[NUM_KEYPOINTS + NUM_IRIS_KEYPOINTS + 2],
              keypoints[NUM_KEYPOINTS + NUM_IRIS_KEYPOINTS + 4]);
          const rightDiameterX = distance(
              keypoints[NUM_KEYPOINTS + NUM_IRIS_KEYPOINTS + 3],
              keypoints[NUM_KEYPOINTS + NUM_IRIS_KEYPOINTS + 1]);

          ctx.beginPath();
          ctx.ellipse(
              rightCenter[0], rightCenter[1], rightDiameterX / 2,
              rightDiameterY / 2, 0, 0, 2 * Math.PI);
          ctx.stroke();
        }
      }
    });
  }

  stats.end();
  rafID = requestAnimationFrame(renderPrediction);
};

async function main() {
  loading('Setting Backend');
  await tf.setBackend(state.backend);
  // setupDatGui();

  stats.showPanel(0);  // 0: fps, 1: ms, 2: mb, 3+: custom
  (document.getElementById('main') as HTMLElement).appendChild(stats.dom);

  loading('Configuring Camera');
  await setupCamera();
  video.play();
  videoWidth = video.videoWidth;
  videoHeight = video.videoHeight;
  video.width = videoWidth;
  video.height = videoHeight;

  canvas = document.getElementById('output') as HTMLCanvasElement;
  canvas.width = videoWidth;
  canvas.height = videoHeight;
  const canvasContainer = document.querySelector('.canvas-wrapper') as HTMLDivElement;
  canvasContainer.setAttribute('style', `width: ${videoWidth}px; height: ${videoHeight}px`);

  ctx = canvas.getContext('2d') as CanvasRenderingContext2D;
  ctx.translate(canvas.width, 0);
  ctx.scale(-1, 1);
  ctx.fillStyle = GREEN;
  ctx.strokeStyle = GREEN;
  ctx.lineWidth = 0.5;

  loading('Loading model');
  model = await faceLandmarksDetection.load(
      faceLandmarksDetection.SupportedPackages.mediapipeFacemesh,
      {maxFaces: state.maxFaces});
      loading(false);
  renderPrediction();
};

function loading(msgOrShow: string | boolean='Loading...') {
  const loadingScreen = document.getElementById('loading');
  if(!msgOrShow) {
    loadingScreen?.style.setProperty('display', 'none');
  } else {
    loadingScreen?.style.setProperty('display', 'flex');
    const msgBox = document.querySelector('#loading .msg') as Element;
    msgBox.textContent = typeof msgOrShow === 'string' ? msgOrShow : 'Loading ...';
  }
}

function calculateLiveliness(predictions:AnnotatedPrediction[]) {
  if(predictions.length > 1) {
    console.log('Multiple persons present')
    return;
  }

  let prediction = predictions[0];
  // @ts-ignore
  let leftEyeLower: Coords3D = prediction.annotations.leftEyeLower0, leftEyeUpper: Coords3D = prediction.annotations.leftEyeUpper0;
  // @ts-ignore
  let rightEyeLower: Coords3D = prediction.annotations.rightEyeLower0, rightEyeUpper: Coords3D = prediction.annotations.rightEyeUpper0;

  let leftEyeLowerMiddle = leftEyeLower[Math.floor(leftEyeLower.length/2)]
  let leftEyeUpperMiddle = leftEyeUpper[Math.floor(leftEyeUpper.length/2)]

  let rightEyeLowerMiddle = rightEyeLower[Math.floor(rightEyeLower.length/2)]
  let rightEyeUpperMiddle = rightEyeUpper[Math.floor(rightEyeUpper.length/2)]

  let leftGap = distance(leftEyeLowerMiddle, leftEyeUpperMiddle);
  let rightGap = distance(rightEyeLowerMiddle, rightEyeUpperMiddle);
  let eyeStateClosed = (leftGap < 5 && rightGap < 5);
  if(eyeStateClosed && state.currentStatus.lastEyeStateOpen) {
    state.currentStatus.blinkCount += 1;
  }
  state.currentStatus.lastEyeStateOpen = !eyeStateClosed;

  if(state.currentStatus.blinkCount >= state.validation.blinkCount) {
    (document.querySelectorAll('.validations li')[0] as Element).className = 'done';
  }
}

(function() {
  const splashScreen = document.getElementById('splash');
  const stageScreen = document.getElementById('stage');
  const startButton = document.getElementById('start');

  startButton?.addEventListener('click', () => {
    splashScreen?.style.setProperty('display', 'none');
    stageScreen?.style.setProperty('display', 'flex');
    main();
  });
})();
