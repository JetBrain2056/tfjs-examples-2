const imageURL = './images/test.jpg';
let model;

let info       = document.getElementById('info');
let prediction = document.getElementById('prediction');
let progress   = document.querySelector('progress');
let image      = document.getElementById('image');

image.src = imageURL;

let CLASSES = [
  'person',
  'bicycle',
  'car',
  'motorbike',
  'aeroplane',
  'bus',
  'train',
  'truck',
  'boat',
  'traffic light',
  'fire hydrant',
  'stop sign',
  'parking meter',
  'bench',
  'bird',
  'cat',
  'dog',
  'horse',
  'sheep',
  'cow',
  'elephant',
  'bear',
  'zebra',
  'giraffe',
  'backpack',
  'umbrella',
  'handbag',
  'tie',
  'suitcase',
  'frisbee',
  'skis',
  'snowboard',
  'sports ball',
  'kite',
  'baseball bat',
  'baseball glove',
  'skateboard',
  'surfboard',
  'tennis racket',
  'bottle',
  'wine glass',
  'cup',
  'fork',
  'knife',
  'spoon',
  'bowl',
  'banana',
  'apple',
  'sandwich',
  'orange',
  'broccoli',
  'carrot',
  'hot dog',
  'pizza',
  'donut',
  'cake',
  'chair',
  'sofa',
  'pottedplant',
  'bed',
  'diningtable',
  'toilet',
  'tvmonitor',
  'laptop',
  'mouse',
  'remote',
  'keyboard',
  'cell phone',
  'microwave',
  'oven',
  'toaster',
  'sink',
  'refrigerator',
  'book',
  'clock',
  'vase',
  'scissors',
  'teddy bear',
  'hair drier',
  'toothbrush',
];

async function postOnServer(data, link) {
  console.log('>>postOnServer()...', link);
  let res;
  try {
      let response = await fetch(link, {
          method  : 'post',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(data)
      });
      res = await response.json();
  } catch (err) {
      console.log(err);
  }
  return res;
}
async function getOnServer(link) {
  console.log('>>getOnServer()...', link);
  let res;
  try{
      let response = await fetch(link);
      res = await response.json();
  } catch (err) {
      console.log(err)
  }
  return res;
}
const trainButton = document.getElementById('train');
trainButton.onclick = async function() {

  const imageURL = './images/train416.jpg';
  image.src = imageURL;

  res = await getOnServer('/tftrain');
  console.log(res);

  if (res.status==='success') {
    info.innerText = 'Model succesfully trained!';  

    await oldPredict(res.data);
    
  } else {
    info.innerText = 'Error!';    
  }
  progress.style.display = 'none';
}
const saveButton = document.getElementById('save');
saveButton.onclick = async function saveModel() {
  console.log('SaveModel...');
  
  try {
    await model.save('downloads://model');      
  } catch(err) {
    console.log(err);
  }
  
}
async function ssd_mobilenet(bufferT) {

  const minScore    = 0.2;
  const maxNumBoxes = 15;

  const imageT = await bufferT.expandDims();

  let modelOut = await model.executeAsync(await tf.cast(imageT, 'int32'));

  let height, width, scores, boxes;

  height = imageT.shape[1];
  width  = imageT.shape[2];
  scores = modelOut[0].dataSync();
  boxes  = modelOut[1].dataSync();

  imageT.dispose();
  tf.dispose(modelOut);
  
  console.log(scores);  
  console.log(boxes);  
  
  function calculateMaxScores(scores, numBoxes, numClasses) {
    let maxes = [];
    let classes = [];
    for (let i = 0; i < numBoxes; i++) {
        let max = Number.MIN_VALUE;
        let index = -1;
        for (let j = 0; j < numClasses; j++) {
            if (scores[i * numClasses + j] > max) {
                max = scores[i * numClasses + j];
                index = j;
            }
        }
        maxes[i]   = max;
        classes[i] = index;
    }
    return [maxes, classes];
  }
  
  const _a = calculateMaxScores(scores, modelOut[0].shape[1], modelOut[0].shape[2]);
  const maxScores = _a[0];
  const classes   = _a[1];

  console.log(maxScores)
  console.log(classes)

  prevBackend = tf.getBackend();
  if (tf.getBackend()==='webgl') {
      tf.setBackend('cpu');
  }
  
  let boxes2  = await tf.tensor2d(boxes, [modelOut[1].shape[1], modelOut[1].shape[3]]);
  // console.log(boxes2);  

  const indexTensor = await tf.image.nonMaxSuppressionAsync(boxes2, maxScores, maxNumBoxes, minScore, minScore);  

  const indexes = await indexTensor.array();
  indexTensor.dispose();  

  tf.setBackend(prevBackend)
  
  const count = indexes.length;  
  console.log(count);  
       
  let objects = [];
  for (let i of indexes) {   

      const minY = boxes[i * 4] * height; 
      const minX = boxes[i * 4 + 1] * width;  
      const maxY = boxes[i * 4 + 2] * height; 
      const maxX = boxes[i * 4 + 3] * width;   
      
      objects.push({
        left  : minX,
        top   : minY,
        right : maxX,
        bottom: maxY,
        className: CLASSES[classes[i]],
        classProb: boxes[i]  
      })
  }
  drawImage(objects);
}
async function oldPredict(predictions) {

  function _logistic(x) {
    if (x > 0) {
        return (1 / (1 + Math.exp(-x)));
    } else {
        const e = Math.exp(x);
        return e / (1 + e);
    }
  }

  // let predictions = await outputs.array();
  // console.log(predictions);  

  const ANCHORS = [0.573, 0.677, 1.87, 2.06, 3.34, 5.47, 7.88, 3.53, 9.77, 9.17]; 

  if (tf.getBackend()==='webgl') {
      tf.setBackend('cpu');
  }
  prevBackend = tf.getBackend();
  // console.log(prevBackend);  

  if (predictions.length != 3) {
		console.log( "Post processing..." );
	
	  const num_anchor = ANCHORS.length / 2;
		const channels   = predictions[0][0][0].length;
		const height     = predictions[0].length;
		const width      = predictions[0][0].length;    
		const num_class  = channels / num_anchor - 5;
    const maxNumBoxes = 20;
    const minIoU      = 0.15;
    const minScore    = 0.05;

    console.log(num_anchor);
    console.log(channels);
    console.log(num_class);

		let boxes   = [];
		let scores  = [];
		let classes = [];

		for (var grid_y = 0; grid_y < height; grid_y++) {
			for (var grid_x = 0; grid_x < width; grid_x++) {
				let offset = 0;
        const pixl = predictions[0][grid_y][grid_x];

        // console.log('pixel:',pixl)    

				for (var i = 0; i < num_anchor; i++) {    

          // console.log('offset:',offset)      
          
					let bx = (_logistic(pixl[offset++]) + grid_x) / width;
					let by = (_logistic(pixl[offset++]) + grid_y) / height;        
					let bw = Math.exp(pixl[offset++]) * ANCHORS[i * 2] / width;
					let bh = Math.exp(pixl[offset++]) * ANCHORS[i * 2 + 1] / height;   
          let objectness = tf.scalar(_logistic(pixl[offset++])); 
					        
					let class_probabilities = tf.tensor1d(pixl.slice(offset, offset + num_class)).softmax();
          
          // console.log(await class_probabilities.array())
          
					offset += num_class;          

					class_probabilities = class_probabilities.mul(objectness);
          // console.log(class_probabilities)
					let max_index       = class_probabilities.argMax();

          // console.log(max_index.dataSync())
          // console.log(max_index.dataSync()[0])
      
          x = bx - bw / 2
          y = by - bh / 2
          w = bw
          h = bh

					boxes.push([x, y, w, h]);
					scores.push(class_probabilities.max().dataSync()[0]);
					classes.push(max_index.dataSync()[0]);
				}
			}
		}

		boxes   = tf.tensor2d(boxes);
		scores  = tf.tensor1d(scores);
		classes = tf.tensor1d(classes);

		const selected_indices = await tf.image.nonMaxSuppressionAsync(boxes, scores, maxNumBoxes, minIoU, minScore);
    // console.log(await selected_indices.array())
    
		predictions = [await boxes.gather(selected_indices).array(), 
                   await scores.gather(selected_indices).array(), 
                   await classes.gather(selected_indices).array()];

	}    

  const boxes   = predictions[0];
  const scores  = predictions[1];
  const classes = predictions[2];  

  console.log(boxes)
  console.log(scores)
  console.log(classes)

  const width    = image.width;
  const height   = image.height;

  let objects = [];
  for (let bbox of boxes) {    

    const indexes = boxes.indexOf(bbox);
    
    const x1 = bbox[0] * width; 
    const y1 = bbox[1] * height;  
    const w1 = bbox[2] * width; 
    const h1 = bbox[3] * height;   
    
    objects.push({
        left  : x1,
        top   : y1,
        right : x1 + w1,
        bottom: y1 + h1,
        className: CLASSES[classes[indexes]],
        classProb: scores[indexes]
    });
  }
  drawImage(objects);
}
function drawImage(objects) {
  console.log(objects); 

  const c = document.getElementById('canvas');
  const context = c.getContext('2d');
  context.drawImage(image,0,0);
  context.font = '12px Arial';

  console.log('Number of detections: ', objects.length);

  context.drawImage(image, 0, 0);
  context.font = '10px Arial';

  for (let obj of objects) {

    x = obj.left;
    y = obj.top;
    w = obj.right - obj.left;
    h = obj.bottom - obj.top;

    context.beginPath();
    context.rect(x,y,w,h);    
    context.lineWidth = 2;
    context.strokeStyle = 'green';
    context.fillStyle   = 'green';
    context.stroke();
    context.fillText(obj.classProb.toFixed(3)+' '+obj.className, x, y);
  }
}
const runButton = document.getElementById('run');
runButton.onclick = async function runPredict() {
  console.log("run predict...");
  
  let bufferT, resizedT, imageT;

  bufferT  = await tf.browser.fromPixels(image);  
  resizedT = await tf.image.resizeNearestNeighbor(bufferT, [416, 416]);  
  imageT   = await resizedT.div(tf.scalar(255.0)).expandDims();  

  //*** Graph Models ***
  // await ssd_mobilenet(bufferT); 
  //*** Layers Models ***  
  outputs = await model.predict(imageT);
  predictions = await outputs.array();
  await oldPredict(predictions);  
}

// Load the image model 
async function init() {
  console.log('Load model...');

  info.innerText = 'Load model. Please wait...';

  //*** Graph Models ***
  // const URL = 'https://storage.googleapis.com/tfjs-models/savedmodel/ssdlite_mobilenet_v2/model.json';
  // const URL = 'https://storage.googleapis.com/tfjs-models/savedmodel/ssd_mobilenet_v2/model.json';  
  // const URL = 'https://storage.googleapis.com/tfjs-models/savedmodel/ssd_mobilenet_v1/model.json';  
  // const URL = 'https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v3_small_100_224/feature_vector/5/default/1';   

  // model = await tf.loadGraphModel(URL);  

  // console.log(model.inputs); 
  // console.log(model.outputs); 

  // let res = await model.executeAsync(await tf.zeros([1, 416, 416, 3], 'int32'));

  // console.log(res); 

  // scores = res[0];
  // boxes  = res[1];

  // console.log(scores);  
  // console.log(boxes);  

  //*** Layers Models ***
  // const URL = 'https://raw.githubusercontent.com/MikeShi42/yolo-tiny-tfjs/master/model2.json';
  // const URL = 'https://storage.googleapis.com/tfjs-examples/simple-object-detection/dist/object_detection_model/model.json';  
  const URL = "./model/model.json";  

  model = await tf.loadLayersModel(URL);  

  // model.summary();

  // console.log(model.inputs); 
  // console.log(model.outputs); 

  info.innerText = 'Model loaded successfully!';
                
}

window.onload = () => init(); 