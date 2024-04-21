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
  const grid_w=7;
  const grid_h=7;
  const cell_w=64;
  const cell_h=64;
  const img_w=grid_w*cell_w;
  const img_h=grid_h*cell_h;
  const kernel_r=tf.regularizers.l2({l2:0.0005});

  const trainModel = tf.sequential();
  
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

  trainModel.add(tf.layers.inputLayer({inputShape :[img_h, img_w, 3]}))
  trainModel.add(tf.layers.conv2d({filters:8,   kernelSize: [3, 3], strides:[2, 2], padding : 'same', activation:'linear'}))
  // trainModel.add(tf.layers.dense({units:64, inputShape :[img_h, img_w, 3]}))
  trainModel.add(tf.layers.flatten())
  // trainModel.add(tf.layers.dense({units:1024}))
  // trainModel.add(tf.layers.dropout({rate:0.5}))
  // trainModel.add(tf.layers.dense({units:128, activation:'sigmoid'}))
  // trainModel.add(tf.layers.dense({units:128}))
  trainModel.add(tf.layers.reshape({targetShape:[7,7,30]}))

  // trainModel.summary();


  const outputModel = tf.sequential();
  outputModel.add(tf.layers.flatten({inputShape :[13, 13, 425]}))  
  outputModel.add(tf.layers.dense({units:64, activation: 'relu' }))
  outputModel.add(tf.layers.dense({units:5}))  

  const newOutput = outputModel.apply(model.outputs[0]);
  model = tf.model({inputs: model.inputs, outputs: newOutput});  

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
  // console.log(imageFeatures);
  trainingDataInputs.push(imageFeatures);
  // output = model.predict(imageFeatures);
  // console.log(output);
  // console.log(output.dataSync());
  // trainingDataOutputs.push(output);

  imageFeatures = await calculateImageFeatures(image2);  
  trainingDataInputs.push(imageFeatures);
  // output = model.predict(imageFeatures);    
  // trainingDataOutputs.push(output);
  
  imageFeatures = await calculateImageFeatures(image3);  
  trainingDataInputs.push(imageFeatures);
  // output = model.predict(imageFeatures);    
  // trainingDataOutputs.push(output);

  imageFeatures = await calculateImageFeatures(image4);   
  trainingDataInputs.push(imageFeatures);
  // output = model.predict(imageFeatures);    
  // trainingDataOutputs.push(output);
  
  console.log(trainingDataInputs);

  gatherDataState = 0;
  trainingDataOutputs.push(gatherDataState);
  trainingDataOutputs.push(gatherDataState);
  gatherDataState = 1;
  trainingDataOutputs.push(gatherDataState);
  trainingDataOutputs.push(gatherDataState);
  console.log(trainingDataOutputs);

  info.innerText = 'Training model. Please wait...';
  // progress.style.display = 'block';
  
  const inputsAsTensor  = await tf.concat(trainingDataInputs);
  // const targetTensor    = await tf.concat(trainingDataOutputs);
  // const targetTensor    = await tf.oneHot(tf.tensor1d(trainingDataOutputs, 'int32'), 2);

  // targetTensor = [];
  // target = tf.tensor1d([0].concat(boundingBox));  
  // targetTensor.push(target);
  // target = tf.tensor1d([0].concat(boundingBox));
  // targetTensor.push(target);
  // target = tf.tensor1d([1].concat(boundingBox));
  // targetTensor.push(target);
  // target = tf.tensor1d([1].concat(boundingBox));
  // targetTensor.push(target);


  console.log(inputsAsTensor);
  console.log(targetTensor);
  
  const epochs = 5;  
  await model.fit(inputsAsTensor, targetTensor, {    
    shuffle   : true, 
    batchSize : 128,     
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

  // bufferT  = await tf.browser.fromPixels(image);  
  // resizedT = await tf.image.resizeNearestNeighbor(bufferT, [244, 244]).toFloat();
  // imageT   = await resizedT.div(tf.scalar(255.0)).expandDims();

  // imageFeatures  = await model.executeAsync(imageT);
  // console.log(imageFeatures);
  
  // let prediction = await trainModel.predict(imageT);

  bufferT  = await tf.browser.fromPixels(image);  
  resizedT = await tf.image.resizeNearestNeighbor(bufferT, [416, 416]);
  imageT   = await resizedT.div(tf.scalar(255.0)).expandDims();
  
  console.log(imageT);

//   if (tf.getBackend()==='webgl') {
//     tf.setBackend('cpu');
// }
   oldPredict(imageT);
  // const outputs = await model.predict(imageT);
  // const outputs = await model.predict(imageT);
  // const modelOut = await model.executeAsync(imageT);
  // console.log(outputs); 
  // console.log(outputs.dataSync()); 

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
async function oldPredict(inputs) {

  const outputs = await model.predict(inputs);
  console.log(outputs.dataSync()); 
	const arrays = !Array.isArray(outputs) ? outputs.array() : Promise.all(outputs.map(t => t.array()));
	let predictions = await arrays;
  console.log(predictions);  

  const ANCHORS = [0.573, 0.677, 1.87, 2.06, 3.34, 5.47, 7.88, 3.53, 9.77, 9.17]; 

  if (predictions.length != 3) {
		console.log( "Post processing..." );
	
	  const num_anchor = ANCHORS.length / 2;
		const channels   = predictions[0][0][0].length;
		const height     = predictions[0].length;
		const width      = predictions[0][0].length;    
		const num_class  = channels / num_anchor - 5;

		let boxes   = [];
		let scores  = [];
		let classes = [];

		for (var grid_y = 0; grid_y < height; grid_y++) {
			for (var grid_x = 0; grid_x < width; grid_x++) {
				let offset = 0;

				for (var i = 0; i < num_anchor; i++) {
					let x = (_logistic(predictions[0][grid_y][grid_x][offset++]) + grid_x) / width;
					let y = (_logistic(predictions[0][grid_y][grid_x][offset++]) + grid_y) / height;
					let w = Math.exp(predictions[0][grid_y][grid_x][offset++]) * ANCHORS[i * 2] / width;
					let h = Math.exp(predictions[0][grid_y][grid_x][offset++]) * ANCHORS[i * 2 + 1] / height;

					let objectness          = tf.scalar(_logistic(predictions[0][grid_y][grid_x][offset++]));
					let class_probabilities = tf.tensor1d(predictions[0][grid_y][grid_x].slice(offset, offset + num_class)).softmax();
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

		const selected_indices = await tf.image.nonMaxSuppressionAsync(boxes, scores, 15);
		predictions = [await boxes.gather(selected_indices).array(), 
                   await scores.gather(selected_indices).array(), 
                   await classes.gather(selected_indices).array()];

	}    
  // console.log(predictions);  
  return predictions;
}
const runButton = document.getElementById('run');
runButton.onclick = async function runPredict() {
  console.log("run predict...");
  
  let bufferT, resizedT, imageT;

  bufferT  = await tf.browser.fromPixels(image);  
  resizedT = await tf.image.resizeNearestNeighbor(bufferT, [416, 416]);
  imageT   = await resizedT.div(tf.scalar(255.0)).expandDims();
  // console.log(imageT);     

  let res = await oldPredict(imageT);
  
  const boxes   = res[0];
  const scores  = res[1];
  const classes = res[2];  

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
function _logistic(x) {
	if (x > 0) {
	    return (1 / (1 + Math.exp(-x)));
	} else {
	    const e = Math.exp(x);
	    return e / (1 + e);
	}
}

// Load the image model 
async function init() {
  console.log('Load model...');

  info.innerText = 'Load model. Please wait...';

  //*** Graph Models ***
  // const URL = 'https://storage.googleapis.com/tfjs-models/savedmodel/ssdlite_mobilenet_v2/model.json';
  // const URL = 'https://storage.googleapis.com/tfjs-models/savedmodel/ssd_mobilenet_v2/model.json';  
  // const URL = 'https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v3_small_100_224/feature_vector/5/default/1';   

  // model = await tf.loadGraphModel(URL);  
  // let res = await model.executeAsync(await tf.zeros([1, 416, 416, 3], 'int32'));

  // console.log(res); 
  // console.log(res.dataSync());

  // scores = res[0];
  // boxes  = res[1];

  // console.log(scores);  
  // console.log(boxes);  

  //*** Layers Models ***
  // const URL = 'https://raw.githubusercontent.com/MikeShi42/yolo-tiny-tfjs/master/model2.json';
  // const URL = 'https://storage.googleapis.com/tfjs-examples/simple-object-detection/dist/object_detection_model/model.json';  
  const URL = "./model/model.json";  

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
