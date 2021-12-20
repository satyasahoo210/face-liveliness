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
  `https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@${tfjsWasm.version_wasm}/dist/`);

const NUM_KEYPOINTS = 468;
const NUM_IRIS_KEYPOINTS = 5;
const GREEN = '#32EEDB';
const RED = '#FF2C35';
const BLUE = '#157AB3';
let stopRendering = false;

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
  },
  face: {
    topLeft: [0, 0] as Coord2D,
    bottomRight: [0, 0] as Coord2D
  },
  eyeGap: {
    left: [] as number[],
    right: [] as number[]
  }
};

async function setupCamera() {
  video = document.getElementById('video') as HTMLVideoElement;

  const stream = await navigator.mediaDevices.getUserMedia({
    'audio': false,
    'video': {
      // 'deviceId': '26ac4c7879bc217604ddb8618960818bb9a7f5ed972a585f0d844c17b8f3c128',
      facingMode: 'user'
    },
  });
  video.srcObject = stream;

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

  const predictions: AnnotatedPrediction[] = await model.estimateFaces({
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
      const keypoints: Coords3D = prediction.scaledMesh as Coords3D;

      // Bounding Box
      // ctx.strokeStyle = GREEN;
      // ctx.beginPath();
      // ctx.rect(topLeft[0], topLeft[1], bottomRight[0], bottomRight[1]);
      // ctx.stroke();

      // ctx.strokeStyle = RED;
      // // @ts-ignore
      // let leftEyeLower = prediction.annotations.leftEyeLower0, leftEyeUpper = prediction.annotations.leftEyeUpper0;
      // // @ts-ignore
      // let rightEyeLower = prediction.annotations.rightEyeLower0, rightEyeUpper = prediction.annotations.rightEyeUpper0;

      // drawPath(ctx, leftEyeUpper, false)
      // drawPath(ctx, leftEyeLower, false)
      // drawPath(ctx, rightEyeUpper, false)
      // drawPath(ctx, rightEyeLower, false)


      // if (keypoints.length > NUM_KEYPOINTS) {
      //   ctx.strokeStyle = RED;
      //   ctx.lineWidth = 1;

      //   const leftCenter = keypoints[NUM_KEYPOINTS];
      //   const leftDiameterY = distance(
      //     keypoints[NUM_KEYPOINTS + 4], keypoints[NUM_KEYPOINTS + 2]);
      //   const leftDiameterX = distance(
      //     keypoints[NUM_KEYPOINTS + 3], keypoints[NUM_KEYPOINTS + 1]);

      //   ctx.beginPath();
      //   ctx.ellipse(
      //     leftCenter[0], leftCenter[1], leftDiameterX / 2, leftDiameterY / 2,
      //     0, 0, 2 * Math.PI);
      //   ctx.stroke();

      // if (keypoints.length > NUM_KEYPOINTS + NUM_IRIS_KEYPOINTS) {
      //   const rightCenter = keypoints[NUM_KEYPOINTS + NUM_IRIS_KEYPOINTS];
      //   const rightDiameterY = distance(
      //     keypoints[NUM_KEYPOINTS + NUM_IRIS_KEYPOINTS + 2],
      //     keypoints[NUM_KEYPOINTS + NUM_IRIS_KEYPOINTS + 4]);
      //   const rightDiameterX = distance(
      //     keypoints[NUM_KEYPOINTS + NUM_IRIS_KEYPOINTS + 3],
      //     keypoints[NUM_KEYPOINTS + NUM_IRIS_KEYPOINTS + 1]);

      //   ctx.beginPath();
      //   ctx.ellipse(
      //     rightCenter[0], rightCenter[1], rightDiameterX / 2,
      //     rightDiameterY / 2, 0, 0, 2 * Math.PI);
      //   ctx.stroke();
      // }
      // }

      // if (true) {
      //   ctx.strokeStyle = BLUE;
      //   ctx.lineWidth = 1;

      //   const leftCenter = keypoints[NUM_KEYPOINTS];
      //   const leftDiameterY = distance(
      //     keypoints[NUM_KEYPOINTS + 4], keypoints[NUM_KEYPOINTS + 2]);
      //   const leftDiameterX = distance(
      //     keypoints[NUM_KEYPOINTS + 3], keypoints[NUM_KEYPOINTS + 1]);

      //   ctx.beginPath();
      //   ctx.ellipse(
      //     leftCenter[0], leftCenter[1], leftDiameterX / 2, leftDiameterY / 2,
      //     0, 0, 2 * Math.PI);
      //   ctx.stroke();

      //   if (keypoints.length > NUM_KEYPOINTS + NUM_IRIS_KEYPOINTS) {
      //     const rightCenter = keypoints[NUM_KEYPOINTS + NUM_IRIS_KEYPOINTS];
      //     const rightDiameterY = distance(
      //       keypoints[NUM_KEYPOINTS + NUM_IRIS_KEYPOINTS + 2],
      //       keypoints[NUM_KEYPOINTS + NUM_IRIS_KEYPOINTS + 4]);
      //     const rightDiameterX = distance(
      //       keypoints[NUM_KEYPOINTS + NUM_IRIS_KEYPOINTS + 3],
      //       keypoints[NUM_KEYPOINTS + NUM_IRIS_KEYPOINTS + 1]);

      //     ctx.beginPath();
      //     ctx.ellipse(
      //       rightCenter[0], rightCenter[1], rightDiameterX / 2,
      //       rightDiameterY / 2, 0, 0, 2 * Math.PI);
      //     ctx.stroke();
      //   }
      // }
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
    { maxFaces: state.maxFaces });
  loading(false);
  renderPrediction();
};

function loading(msgOrShow: string | boolean = 'Loading...') {
  const loadingScreen = document.getElementById('loading');
  if (!msgOrShow) {
    loadingScreen?.style.setProperty('display', 'none');
  } else {
    loadingScreen?.style.setProperty('display', 'flex');
    const msgBox = document.querySelector('#loading .msg') as Element;
    msgBox.textContent = typeof msgOrShow === 'string' ? msgOrShow : 'Loading ...';
  }
}

function calculateLiveliness(predictions: AnnotatedPrediction[]) {
  if (predictions.length > 1) {
    console.log('Multiple persons present')
    return;
  }

  let prediction = predictions[0];
  // @ts-ignore
  let leftEyeLower: Coords3D = prediction.annotations.leftEyeLower0, leftEyeUpper: Coords3D = prediction.annotations.leftEyeUpper0;
  // @ts-ignore
  let rightEyeLower: Coords3D = prediction.annotations.rightEyeLower0, rightEyeUpper: Coords3D = prediction.annotations.rightEyeUpper0;

  let leftEyeLowerMiddle = leftEyeLower[Math.floor(leftEyeLower.length / 2)]
  let leftEyeUpperMiddle = leftEyeUpper[Math.floor(leftEyeUpper.length / 2)]

  let rightEyeLowerMiddle = rightEyeLower[Math.floor(rightEyeLower.length / 2)]
  let rightEyeUpperMiddle = rightEyeUpper[Math.floor(rightEyeUpper.length / 2)]

  let leftGap = distance(leftEyeLowerMiddle, leftEyeUpperMiddle);
  let rightGap = distance(rightEyeLowerMiddle, rightEyeUpperMiddle);

  state.eyeGap.left.push(leftGap);
  state.eyeGap.right.push(rightGap);

  if (state.eyeGap.left.length > 10) {
    state.eyeGap.left.splice(0, 1)
  }

  if (state.eyeGap.right.length > 10) {
    state.eyeGap.right.splice(0, 1)
  }

  let leftAvgGap = state.eyeGap.left.reduce((a, c) => a + c) / state.eyeGap.left.length
  let rightAvgGap = state.eyeGap.right.reduce((a, c) => a + c) / state.eyeGap.right.length

  let leftDiff = leftAvgGap - leftGap;
  let rightDiff = rightAvgGap - rightGap;

  let eyeStateClosed = (leftDiff > 5 && rightDiff > 5);

  // console.log(eyeStateClosed, leftDiff, rightDiff);
  if (document.querySelectorAll('.validations>li li')[state.currentStatus.blinkCount] && eyeStateClosed && state.currentStatus.lastEyeStateOpen) {
    state.currentStatus.blinkCount += 1;
    (document.querySelectorAll('.validations>li li')[state.currentStatus.blinkCount - 1].querySelector('input') as HTMLInputElement).checked = true;
  }
  state.currentStatus.lastEyeStateOpen = !eyeStateClosed;

  if (state.currentStatus.blinkCount >= state.validation.blinkCount) {
    (document.querySelectorAll('.validations>li')[0] as Element).className = 'done';
  }

  state.face.topLeft = prediction.boundingBox.topLeft as Coord2D;
  state.face.bottomRight = prediction.boundingBox.bottomRight as Coord2D;
}

(function () {
  const splashScreen = document.getElementById('splash');
  const stageScreen = document.getElementById('stage');
  const startButton = document.getElementById('start');
  const liveFaceNextButton = document.getElementById('liveface-next') as HTMLButtonElement;
  const imageFile = document.getElementById('image2') as HTMLInputElement;

  const faceMatch = document.getElementById('faceMatch') as HTMLElement;
  const ocr = document.getElementById('ocr') as HTMLElement;

  imageFile.onchange = function (ev: Event) {
    if (imageFile.files?.length as number > 0) {
      liveFaceNextButton.disabled = false;
    }
  }

  startButton?.addEventListener('click', () => {
    splashScreen?.style.setProperty('display', 'none');
    stageScreen?.style.setProperty('display', 'flex');
    main();
  });
  liveFaceNextButton?.addEventListener('click', () => {
    // let { topLeft, bottomRight } = state.face
    // let faceImage = ctx.getImageData(topLeft[0], topLeft[1], bottomRight[0], bottomRight[1]);

    // let canvas = document.createElement('canvas');
    // let _ctx = canvas.getContext('2d') as CanvasRenderingContext2D;
    // canvas.width = faceImage.width;
    // canvas.height = faceImage.height;
    // _ctx.putImageData(faceImage, 0, 0);

    let base64 = canvas.toDataURL('image/jpeg');
    console.log(base64)
    let file = dataURLtoFile(base64, 'face.jpg')

    // document.body.appendChild(canvas)
    // match

    let data = new FormData();
    data.append('image', file, file.name);

    request('https://ankitpitroda.com:5000/upload', 'POST', function () {
      if (this.readyState == 4 && this.status == 200) {
        let res = typeof this.response === 'string' ? JSON.parse(this.response) : this.response
        const { path: img_1 } = res;

        file = (imageFile.files as FileList)[0]
        let data2 = new FormData();
        data2.append('image', file, file.name);
        request('https://ankitpitroda.com:5000/upload', 'POST', function () {
          let res2 = typeof this.response === 'string' ? JSON.parse(this.response) : this.response
          const { path: img_2 } = res2;

          let data3 = new FormData();
          data3.append('img_1', img_1);
          data3.append('img_2', img_2);
          request('https://ankitpitroda.com:5000/face_match', 'POST', function () {
            let res3 = typeof this.response === 'string' ? JSON.parse(this.response) : this.response
            // console.log(res3)
            faceMatch.innerText = `Face Match: ${res3.face_match}`

            let data4 = new FormData();
            data4.append('img_1', img_2);
            request('https://ankitpitroda.com:5000/ocr', 'POST', function () {
              let res4 = typeof this.response === 'string' ? JSON.parse(this.response) : this.response
              // console.log(res4)
              ocr.innerText = JSON.stringify(res4, null, 2);
            }, data4)
          }, data3)
        }, data2)
      }
    }, data)

  });
})();

const dataURLtoFile = (dataurl: string, filename: string) => {
  let arr = dataurl.split(','),
    mime = (arr[0].match(/:(.*?);/) as RegExpMatchArray)[1],
    bstr = atob(arr[1]),
    n = bstr.length,
    u8arr = new Uint8Array(n);
  while (n--) {
    u8arr[n] = bstr.charCodeAt(n);
  }
  return new File([u8arr], filename, { type: mime });
};

const request = (
  url: string | URL,
  method = "POST",
  cb: ((this: XMLHttpRequest, ev: Event) => any) | null,
  data: string | Document | ArrayBufferView | ArrayBuffer | Blob | FormData | null | undefined
) => {
  var xhttp = new XMLHttpRequest();
  xhttp.onreadystatechange = cb;
  xhttp.open(method, url, true);
  // xhttp.setRequestHeader("Content-type", "multipart/form-data");
  xhttp.setRequestHeader("Accept", 'application/json');
  xhttp.setRequestHeader('Accept-Language', 'en-US,en;q=0.8');
  xhttp.send(data);
  return xhttp;
}
