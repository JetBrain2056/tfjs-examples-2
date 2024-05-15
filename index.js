const imageURL = './images/test.jpg';
let model;

let info       = document.getElementById('info');
let prediction = document.getElementById('prediction');
let progress   = document.querySelector('progress');
let image      = document.getElementById('image');

image.src = imageURL;

const CLASSES = [
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

const trainButton = document.getElementById('train');
trainButton.onclick = async function() {

  // const nb_boxes=1;
  // const grid_w=7;
  // const grid_h=7;
  // const cell_w=64;
  // const cell_h=64;
  // const img_w=grid_w*cell_w;
  // const img_h=grid_h*cell_h;
  // const kernel_r=tf.regularizers.l2({l2:0.0005});

  // const trainModel = tf.sequential();
  
  // trainModel.add(tf.layers.conv2d({filters:256, kernelSize: [7, 7], strides:[1, 1], inputShape :[img_h, img_w, 3], activation:'relu'}))
  // trainModel.add(tf.layers.maxPooling2d({poolSize:[2, 2], strides:[2, 2], padding : 'same'}))

  // trainModel.add(tf.layers.conv2d({filters:192, kernelSize: [3, 3], padding : 'same', activation:'relu', kernelRegularizer:kernel_r}))
  // trainModel.add(tf.layers.maxPooling2d({poolSize:[2, 2], strides:[2, 2], padding : 'same'}))
  // trainModel.add(tf.layers.conv2d({filters:128, kernelSize: [1, 1], padding : 'same', activation:'relu', kernelRegularizer:kernel_r}))
  // trainModel.add(tf.layers.conv2d({filters:256, kernelSize: [3, 3], padding : 'same', activation:'relu', kernelRegularizer:kernel_r}))
  // trainModel.add(tf.layers.conv2d({filters:256, kernelSize: [1, 1], padding : 'same', activation:'relu', kernelRegularizer:kernel_r}))
  // trainModel.add(tf.layers.conv2d({filters:512, kernelSize: [3, 3], padding : 'same', activation:'relu', kernelRegularizer:kernel_r}))
  // trainModel.add(tf.layers.maxPooling2d({poolSize:[2, 2], strides:[2, 2], padding : 'same'}))

  // trainModel.add(tf.layers.conv2d({filters:256, kernelSize: [1, 1], padding : 'same', activation:'relu', kernelRegularizer:kernel_r}))
  // trainModel.add(tf.layers.conv2d({filters:512, kernelSize: [3, 3], padding : 'same', activation:'relu', kernelRegularizer:kernel_r}))
  // trainModel.add(tf.layers.conv2d({filters:256, kernelSize: [1, 1], padding : 'same', activation:'relu', kernelRegularizer:kernel_r}))
  // trainModel.add(tf.layers.conv2d({filters:512, kernelSize: [3, 3], padding : 'same', activation:'relu', kernelRegularizer:kernel_r}))
  // trainModel.add(tf.layers.conv2d({filters:256, kernelSize: [1, 1], padding : 'same', activation:'relu', kernelRegularizer:kernel_r}))
  // trainModel.add(tf.layers.conv2d({filters:512, kernelSize: [3, 3], padding : 'same', activation:'relu', kernelRegularizer:kernel_r}))
  // trainModel.add(tf.layers.conv2d({filters:256, kernelSize: [1, 1], padding : 'same', activation:'relu', kernelRegularizer:kernel_r}))
  // trainModel.add(tf.layers.conv2d({filters:512, kernelSize: [3, 3], padding : 'same', activation:'relu', kernelRegularizer:kernel_r}))
  // trainModel.add(tf.layers.conv2d({filters:512, kernelSize: [1, 1], padding : 'same', activation:'relu', kernelRegularizer:kernel_r}))
  // trainModel.add(tf.layers.conv2d({filters:1024, kernelSize: [3, 3], padding : 'same', activation:'relu', kernelRegularizer:kernel_r}))
  // trainModel.add(tf.layers.maxPooling2d({poolSize:[2, 2], strides:[2, 2], padding : 'same'}))

  // trainModel.add(tf.layers.conv2d({filters:512, kernelSize: [1, 1], padding : 'same', activation:'relu', kernelRegularizer:kernel_r}))
  // trainModel.add(tf.layers.conv2d({filters:1024, kernelSize: [3, 3], padding : 'same', activation:'relu', kernelRegularizer:kernel_r}))
  // trainModel.add(tf.layers.conv2d({filters:512, kernelSize: [1, 1], padding : 'same', activation:'relu', kernelRegularizer:kernel_r}))
  // trainModel.add(tf.layers.conv2d({filters:1024, kernelSize: [3, 3], padding : 'same', activation:'relu', kernelRegularizer:kernel_r}))
  // trainModel.add(tf.layers.conv2d({filters:1024, kernelSize: [3, 3], padding : 'same', activation:'relu', kernelRegularizer:kernel_r}))
  // trainModel.add(tf.layers.conv2d({filters:1024, kernelSize: [3, 3], strides:[2, 2], padding : 'same'}))

  // trainModel.add(tf.layers.conv2d({filters:1024, kernelSize: [3, 3], activation:'relu', kernelRegularizer:kernel_r}))
  // trainModel.add(tf.layers.conv2d({filters:1024, kernelSize: [3, 3], activation:'relu', kernelRegularizer:kernel_r}))

  // trainModel.add(tf.layers.inputLayer({inputShape :[img_h, img_w, 3]}))
  // trainModel.add(tf.layers.conv2d({filters:8,   kernelSize: [3, 3], strides:[2, 2], padding : 'same', activation:'linear'}))
  // trainModel.add(tf.layers.dense({units:64, inputShape :[img_h, img_w, 3]}))
  // trainModel.add(tf.layers.flatten())
  // trainModel.add(tf.layers.dense({units:1024}))
  // trainModel.add(tf.layers.dropout({rate:0.5}))
  // trainModel.add(tf.layers.dense({units:128, activation:'sigmoid'}))
  // trainModel.add(tf.layers.dense({units:128}))
  // trainModel.add(tf.layers.reshape({targetShape:[7,7,30]}))

  // trainModel.summary();

  // const outputModel = tf.sequential();
  // outputModel.add(tf.layers.flatten({inputShape :[13, 13, 425]}))  
  // outputModel.add(tf.layers.dense({units:64, activation: 'relu' }))
  // outputModel.add(tf.layers.dense({units:4}))  

  // const newOutput = outputModel.apply(model.outputs[0]);
  // model = tf.model({inputs: model.inputs, outputs: newOutput});  

  const optimizer = tf.train.adam();
  model.compile({
    optimizer: optimizer,
    loss: 'binaryCrossentropy',
    metrics: ['accuracy'],
  });

  let trainingDataInputs  = [];
  let trainingDataOutputs = [];

  let bufferT, resizedT, imageT, imageFeatures;

  async function calculateImageFeatures(_image) {
    const imageSize = 416;
       
    bufferT  = await tf.browser.fromPixels(_image);      
    resizedT = await tf.image.resizeNearestNeighbor(bufferT, [imageSize, imageSize]).toFloat();
    imageT   = await resizedT.div(tf.scalar(255.0)).expandDims();

    return imageT;    
  }
  
  imageFeatures = await calculateImageFeatures(image1);    
  trainingDataInputs.push(imageFeatures);
  imageFeatures = await calculateImageFeatures(image2);  
  trainingDataInputs.push(imageFeatures);
  imageFeatures = await calculateImageFeatures(image3);  
  trainingDataInputs.push(imageFeatures);  
  imageFeatures = await calculateImageFeatures(image4);   
  trainingDataInputs.push(imageFeatures);    
  
  console.log(trainingDataInputs);

  info.innerText = 'Training model. Please wait...';
  // progress.style.display = 'block';
  
  const inputsAsTensor  = await tf.concat(trainingDataInputs);  
	
 // inputsAsTensor = tf.zeros([1,416,416,3]);  
 // targetTensor = tf.zeros([4,13,13,425]);

//
  let img = [];
  img[0] = {class:0, xmin:0, ymin:0, xmax:62, ymax:209}
  img[1] = {class:0, xmin:0, ymin:0, xmax:69, ymax:211}  
  img[2] = {class:1, xmin:0, ymin:0, xmax:59, ymax:180}
  img[3] = {class:1, xmin:0, ymin:0, xmax:62, ymax:181}
  
  for (let c of img) {
    pixl = [];    
    for (let xx=0; xx<169; xx++) {    
      for (let i=0; i<5; i++) {
     
        _x = 0
        _y = 0
        _w = 0
        _h = 0
        o = 0
        pixl.push(_x)
        pixl.push(_y)
        pixl.push(_w)
        pixl.push(_h)
        pixl.push(o)
        
        for (let ii=0; ii<80; ii++) {                               
            pixl.push(0)
        }
      }      
    }   
    
    const ANCHORS = [0.573, 0.677, 1.87, 2.06, 3.34, 5.47, 7.88, 3.53, 9.77, 9.17]; 
    function _alogistic(e) {      
      // if (e===0||e<0) {
      //   return Math.log(e - 1/e)
      // } else {
      //   return Math.log(1 - 1/e)
      // }
      return Math.log(e)
    }

    //Normalize in 0~1 and transfer to feature map size:
    const x = c.xmin/416*13;
    const y = c.ymin/416*13;
    const w = (c.xmax - c.xmin)/416*13;
    const h = (c.ymax - c.ymin)/416*13;

    // console.log(x)
    // console.log(y)
    // console.log(w)
    // console.log(h)

    //Normalize coordinates to scores example
    // x1 = 0.010362043045461178  => -1.097084879875183 
    // y1 = -0.013428867794573307 => -0.6710431575775146
    // w1 = 0.18198972940444946 => -0.40344923734664917
    // h1 = 0.5270078182220459  => 0.25029298663139343 

    // console.log((x + w/2) - 1)
    // console.log((y + h/2) - 3)    
    // console.log(Math.log(0.18*13/ANCHORS[2 * 2]))
    // console.log(Math.log(0.52*13/ANCHORS[2 * 2 + 1]))

    let offset = 0;
    for (xg = 0; xg<13; xg++) {
      for (yg = 0; yg<13; yg++) {        
        for (p = 0; p<5; p++) {
          //The center of the trained image
          if (xg===1&&yg===3&&(p===2)) { 
            // Transfer to 0~1 corresponding to each grid cell:
            pixl[offset++] = _alogistic(x + w/2) - xg
            pixl[offset++] = _alogistic(y + h/2) - yg
            pixl[offset++] = Math.log(w/ANCHORS[p * 2])
            pixl[offset++] = Math.log(h/ANCHORS[p * 2 + 1])      
            // pixl[offset++] = (x + w/2) - xg
            // pixl[offset++] = (y + h/2) - yg
            // pixl[offset++] = Math.log(w/ANCHORS[p * 2])
            // pixl[offset++] = Math.log(h/ANCHORS[p * 2 + 1])      
            pixl[offset++] = 1

            if (c.class===0) {                                
              pixl[offset]=1
            } else if (c.class===1) {           
              pixl[offset+1]=1
            }      
          } else {       
            offset += 5;
          }
          offset += 80;
        }
      }
    }
    
    trainingDataOutputs.push(await tf.tensor(pixl, [1,13,13,425], 'float32'));
  }
    
  targetTensor    = await tf.concat(trainingDataOutputs);  
  console.log(targetTensor.dataSync())
	
  console.log(inputsAsTensor);
  console.log(targetTensor);
  
  const epochs = 5;  
  await model.fit(inputsAsTensor, targetTensor, {    
    shuffle   : true, 
    batchSize : 64,     
    epochs    : epochs, 
    callbacks : { onEpochEnd: async (epoch,logs) => {
      progress.value = epoch/(epochs-1)*100;
      console.log('Epoch', epoch, logs)
    }}
  });

  // Freeze the layers of the pre-trained model except for the last few
  for (let i = 0; i < model.layers.length; i++) {
    model.layers[i].trainable = true;
  }

  inputsAsTensor.dispose();
  targetTensor.dispose();  

  // model.dispose();

  info.innerText = 'Model succesfully trained!';
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

function _logistic(x) {
	if (x > 0) {
	    return (1 / (1 + Math.exp(-x)));
	} else {
	    const e = Math.exp(x);
	    return e / (1 + e);
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

  const indexes = indexTensor.dataSync();
  indexTensor.dispose();  

  tf.setBackend(prevBackend)
  
  const count = indexes.length;  
  console.log(count);  
       
  let objects = [];
  for (let i = 0; i < count; i++) {

      const bbox = []
      for (let j = 0; j < 4; j++) {
        bbox[j] = boxes[indexes[i] * 4 + j]
      }

      const minY = bbox[0] * height; 
      const minX = bbox[1] * width;  
      const maxY = bbox[2] * height; 
      const maxX = bbox[3] * width;   
      
      objects.push({
        left  : minX,
        top   : minY,
        right : maxX,
        bottom: maxY,
        className: CLASSES[classes[indexes[i]]],
        classProb: boxes[indexes[i]]  
      })
  }
  drawImage(objects);
}

async function oldPredict(inputs) {

  const outputs = await model.predict(inputs);

  console.log(outputs); 
  let predictions = await outputs.array();
  console.log(predictions);  

  const ANCHORS = [0.573, 0.677, 1.87, 2.06, 3.34, 5.47, 7.88, 3.53, 9.77, 9.17]; 

  if (tf.getBackend()==='webgl') {
      tf.setBackend('cpu');
  }
  prevBackend = tf.getBackend();
  console.log(prevBackend);  

  if (predictions.length != 3) {
		console.log( "Post processing..." );
	
	  const num_anchor = ANCHORS.length / 2;
		const channels   = predictions[0][0][0].length;
		const height     = predictions[0].length;
		const width      = predictions[0][0].length;    
		const num_class  = channels / num_anchor - 5;
    const maxNumBoxes = 15;

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

				for (var i = 0; i < num_anchor; i++) {          
          
					let x = (_logistic(pixl[offset++]) + grid_x) / width;
					let y = (_logistic(pixl[offset++]) + grid_y) / height;
					let w = Math.exp(pixl[offset++]) * ANCHORS[i * 2] / width;
					let h = Math.exp(pixl[offset++]) * ANCHORS[i * 2 + 1] / height;

					let objectness          = tf.scalar(_logistic(pixl[offset++]));
					let class_probabilities = tf.tensor1d(pixl.slice(offset, offset + num_class)).softmax();
					offset += num_class;

					class_probabilities = class_probabilities.mul(objectness);
					let max_index       = class_probabilities.argMax();

					boxes.push([x - w / 2, y - h / 2, x + w / 2, y + h / 2]);
					scores.push(class_probabilities.max().dataSync()[0]);
					classes.push(max_index.dataSync()[0]);
				}
			}
		}

		boxes   = tf.tensor2d(boxes);
		scores  = tf.tensor1d(scores);
		classes = tf.tensor1d(classes);

		const selected_indices = await tf.image.nonMaxSuppressionAsync(boxes, scores, maxNumBoxes);
    console.log(selected_indices)
		predictions = [await boxes.gather(selected_indices).array(), 
                   await scores.gather(selected_indices).array(), 
                   await classes.gather(selected_indices).array()];

	}    

  const boxes   = predictions[0];
  const scores  = predictions[1];
  const classes = predictions[2];  

  const width    = image.width;
  const height   = image.height;

  let objects = [];
  for (let bbox of boxes) {    

    const indexes = boxes.indexOf(bbox);
    
    const minX = bbox[0] * width; 
    const minY = bbox[1] * height;  
    const maxX = bbox[2] * width; 
    const maxY = bbox[3] * height;   
    
    objects.push({
        left  : minX,
        top   : minY,
        right : maxX,
        bottom: maxY,
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

  // console.log(imageT);     

  //*** Graph Models ***
  // await ssd_mobilenet(bufferT); 
  //*** Layers Models ***  
  await oldPredict(imageT);  
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
  const URL = 'https://raw.githubusercontent.com/MikeShi42/yolo-tiny-tfjs/master/model2.json';
  // const URL = 'https://storage.googleapis.com/tfjs-examples/simple-object-detection/dist/object_detection_model/model.json';  
  // const URL = "./model/model.json";  

  model = await tf.loadLayersModel(URL);  

  model.summary();

  // console.log(model.inputs); 
  // console.log(model.outputs); 

  // let res = await model.predict(await tf.zeros([1, 416, 416, 3]));

  // console.log(res); 
  // console.log(res.dataSync());

  info.innerText = 'Model loaded successfully!';
                
}

window.onload = () => init(); 
