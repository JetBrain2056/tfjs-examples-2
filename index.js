const imageURL = './images/test.jpg';
let model;

let info       = document.getElementById('info');
let prediction = document.getElementById('prediction');
let progress   = document.querySelector('progress');
let image      = document.getElementById('image');

image.src = imageURL;
image1.src = './images/C1_train1.jpg';
image2.src = './images/C1_train2.jpg';
image3.src = './images/C2_train1.jpg';
image4.src = './images/C2_train2.jpg';

// let class_names = [];
// class_names.push('Class1');
// class_names.push('Class2');

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
async function yolo_tiny(input) {

  const DEFAULT_INPUT_DIM = 416;
  const DEFAULT_MAX_BOXES = 2048; 
  const DEFAULT_FILTER_BOXES_THRESHOLD = 0.01;
  const DEFAULT_IOU_THRESHOLD = 0.4;
  const DEFAULT_CLASS_PROB_THRESHOLD = 0.5
  const DEFAULT_MODEL_LOCATION = 'https://raw.githubusercontent.com/MikeShi42/yolo-tiny-tfjs/master/model2.json';

  const YOLO_ANCHORS = tf.tensor2d([
    [0.57273, 0.677385], [1.87446, 2.06253], [3.33843, 5.47434],
    [7.88282, 3.52778], [9.77052, 9.16828],
  ]);

  results = await yolo(input);

  console.log(results);

  drawImage(results);

  async function yolo(
    input,    
    {
      classProbThreshold   = DEFAULT_CLASS_PROB_THRESHOLD,
      iouThreshold         = DEFAULT_IOU_THRESHOLD,
      filterBoxesThreshold = DEFAULT_FILTER_BOXES_THRESHOLD,
      yoloAnchors          = YOLO_ANCHORS,
      maxBoxes             = DEFAULT_MAX_BOXES,
      width: widthPx       = DEFAULT_INPUT_DIM,
      height: heightPx     = DEFAULT_INPUT_DIM,
      numClasses = 80,
      classNames = class_names,
    } = {},
  ) {
    let activation = await model.predict(input);

    const outs = tf.tidy(() => { // Keep as one var to dispose easier
    
      const [box_xy, box_wh, box_confidence, box_class_probs ] =
        yolo_head(activation, yoloAnchors, numClasses);

      const all_boxes = yolo_boxes_to_corners(box_xy, box_wh);

      let [boxes, scores, classes] = yolo_filter_boxes(
        all_boxes, box_confidence, box_class_probs, filterBoxesThreshold);

      // If all boxes have been filtered out
      if (boxes == null) {
        return null;
      }

      const width = tf.scalar(widthPx);
      const height = tf.scalar(heightPx);

      const image_dims = tf.stack([height, width, height, width]).reshape([1,4]);

      boxes = tf.mul(boxes, image_dims);

      return [boxes, scores, classes];
    });

    if (outs === null) {
      return [];
    }

    const [boxes, scores, classes] = outs;

    const indices = await tf.image.nonMaxSuppressionAsync(boxes, scores, maxBoxes, iouThreshold)

    // Pick out data that wasn't filtered out by NMS and put them into
    // CPU land to pass back to consumer
    const classes_indx_arr = await classes.gather(indices).data();
    const keep_scores      = await scores.gather(indices).data();
    const boxes_arr        = await boxes.gather(indices).data();

    tf.dispose(outs);
    indices.dispose();

    const results = [];

    classes_indx_arr.forEach((class_indx, i) => {
      const classProb = keep_scores[i];
      if (classProb < classProbThreshold) {
        return;
      }

      const className = classNames[class_indx];
      let [top, left, bottom, right] = [
        boxes_arr[4 * i],
        boxes_arr[4 * i + 1],
        boxes_arr[4 * i + 2],
        boxes_arr[4 * i + 3],
      ];

      top    = Math.max(0, top);
      left   = Math.max(0, left);
      bottom = Math.min(heightPx, bottom);
      right  = Math.min(widthPx, right);

      const resultObj = {
        className,
        classProb,
        bottom,
        top,
        left,
        right,
      };

      results.push(resultObj);
    });

    return results;
  }
  function yolo_filter_boxes(
    boxes,
    box_confidence,
    box_class_probs,
    threshold
  ) {
    const box_scores = tf.mul(box_confidence, box_class_probs);
    const box_classes = tf.argMax(box_scores, -1);
    const box_class_scores = tf.max(box_scores, -1);

    const prediction_mask = tf.greaterEqual(box_class_scores, tf.scalar(threshold)).as1D();

    const N = prediction_mask.size
    // linspace start/stop is inclusive.
    const all_indices = tf.linspace(0, N - 1, N).toInt();
    const neg_indices = tf.zeros([N], 'int32');
    const indices = tf.where(prediction_mask, all_indices, neg_indices);

    return [
      tf.gather(boxes.reshape([N, 4]), indices),
      tf.gather(box_class_scores.flatten(), indices),
      tf.gather(box_classes.flatten(), indices),
    ];
  }
  function yolo_boxes_to_corners(box_xy, box_wh) {
    const two = tf.tensor1d([2.0]);
    const box_mins = tf.sub(box_xy, tf.div(box_wh, two));
    const box_maxes = tf.add(box_xy, tf.div(box_wh, two));

    const dim_0 = box_mins.shape[0];
    const dim_1 = box_mins.shape[1];
    const dim_2 = box_mins.shape[2];
    const size = [dim_0, dim_1, dim_2, 1];

    return tf.concat([
      box_mins.slice([0, 0, 0, 1], size),
      box_mins.slice([0, 0, 0, 0], size),
      box_maxes.slice([0, 0, 0, 1], size),
      box_maxes.slice([0, 0, 0, 0], size),
    ], 3);
  }
  function yolo_head(feats, anchors, num_classes) {
    const num_anchors = anchors.shape[0];

    const anchors_tensor = tf.reshape(anchors, [1, 1, num_anchors, 2]);

    let conv_dims = feats.shape.slice(1, 3);

    // For later use
    const conv_dims_0 = conv_dims[0];
    const conv_dims_1 = conv_dims[1];

    let conv_height_index = tf.range(0, conv_dims[0]);
    let conv_width_index = tf.range(0, conv_dims[1]);
    conv_height_index = tf.tile(conv_height_index, [conv_dims[1]])

    conv_width_index = tf.tile(tf.expandDims(conv_width_index, 0), [conv_dims[0], 1]);
    conv_width_index = tf.transpose(conv_width_index).flatten();

    let conv_index = tf.transpose(tf.stack([conv_height_index, conv_width_index]));
    conv_index = tf.reshape(conv_index, [conv_dims[0], conv_dims[1], 1, 2])
    conv_index = tf.cast(conv_index, feats.dtype);

    feats = tf.reshape(feats, [conv_dims[0], conv_dims[1], num_anchors, num_classes + 5]);
    conv_dims = tf.cast(tf.reshape(tf.tensor1d(conv_dims), [1,1,1,2]), feats.dtype);

    let box_xy = tf.sigmoid(feats.slice([0,0,0,0], [conv_dims_0, conv_dims_1, num_anchors, 2]))
    let box_wh = tf.exp(feats.slice([0,0,0, 2], [conv_dims_0, conv_dims_1, num_anchors, 2]))
    const box_confidence = tf.sigmoid(feats.slice([0,0,0, 4], [conv_dims_0, conv_dims_1, num_anchors, 1]))
    const box_class_probs = tf.softmax(feats.slice([0,0,0, 5],[conv_dims_0, conv_dims_1, num_anchors, num_classes]));

    box_xy = tf.div(tf.add(box_xy, conv_index), conv_dims);
    box_wh = tf.div(tf.mul(box_wh, anchors_tensor), conv_dims);

    return [ box_xy, box_wh, box_confidence, box_class_probs ];
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

  // await yolo_tiny(imageT);

  return;

  height = imageT.shape[1];
  width  = imageT.shape[2];
  scores = modelOut[0].dataSync();
  boxes  = modelOut[1].dataSync();

  imageT.dispose();
  tf.dispose(modelOut);

  minScore  = 0.4;
  maxNumBoxes = 30;
  
  console.log(scores);  
  console.log(boxes);  
     
  if (tf.getBackend()==='webgl') {
      tf.setBackend('cpu');
  }
  prevBackend = tf.getBackend();
  console.log(prevBackend);  
  
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
  
  _a = calculateMaxScores(scores, modelOut[0].shape[1], modelOut[0].shape[2]);
  maxScores = _a[0];
  classes   = _a[1];
  
  let boxes2  = await tf.tensor2d(boxes, [modelOut[1].shape[1], modelOut[1].shape[3]]);
  indexTensor = await tf.image.nonMaxSuppressionAsync(boxes2, maxScores, maxNumBoxes, minScore, minScore);  

  indexes = indexTensor.dataSync();
  indexTensor.dispose();  
  
  const count = indexes.length;  
  console.log(count);  
  
  let objects2 = [];
  for (let i = 0; i < count; i++) {
        let bbox = [];
        for (let j = 0; j < 4; j++) {
            bbox[j] = boxes[indexes[i] * 4 + j];
        }
        const minY = bbox[0] * height; 
        const minX = bbox[1] * width;  
        const maxY = bbox[2] * height; 
        const maxX = bbox[3] * width;  
        bbox[0] = minX;  
        bbox[1] = minY;  
        bbox[2] = maxX - minX;  
        bbox[3] = maxY - minY;  
        objects2.push({
            bbox: bbox,
            class: CLASSES[classes[indexes[i]] + 1].displayName,
            score: boxes[indexes[i]]
        });
  }
  // drawImage(objects2);
 
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

  for (let i = 0; i < objects.length; i++) {

    x = objects[i].left;
    y = objects[i].top;
    w = objects[i].right - objects[i].left;
    h = objects[i].bottom - objects[i].top;

    context.beginPath();
    context.rect(x,y,w,h);    
    context.lineWidth = 2;
    context.strokeStyle = 'green';
    context.fillStyle   = 'green';
    context.stroke();
    context.fillText(objects[i].classProb.toFixed(3)+' '+objects[i].className, x, y);
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