const fs   = require('fs');
// const path = require('path');
// const tf   = require("@tensorflow/tfjs");
const tf = require('@tensorflow/tfjs-node');
require('@tensorflow/tfjs-node-gpu');
// const URL   = 'http://localhost:9000/app/model/model.json';
const URL = 'https://raw.githubusercontent.com/MikeShi42/yolo-tiny-tfjs/master/model2.json';

let model;

function dateNow() {
    let date = new Date(Date.now() - (new Date()).getTimezoneOffset() * 60000);
    return date.toISOString().slice(0, 19).replace('T', ' ');
}
function trainModel() {

    // const nb_boxes=1;
    // const grid_w=13;
    // const grid_h=13;
    // const cell_w=32;
    // const cell_h=32;
    // const img_w=grid_w*cell_w;
    // const img_h=grid_h*cell_h;
    // const kernel_r=tf.regularizers.l2({l2:0.0005});

    // const trainModel = tf.sequential();
    
    // trainModel.add(tf.layers.conv2d({filters:256, kernelSize: [3, 3], strides:[1, 1], inputShape :[img_h, img_w, 3], activation:'relu'}))
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

    // trainModel.add(tf.layers.flatten())
    // trainModel.add(tf.layers.dense({units:1024}))
    // trainModel.add(tf.layers.dropout({rate:0.5}))
    // trainModel.add(tf.layers.dense({units:128, activation:'sigmoid'}))
    // trainModel.add(tf.layers.dense({units:128}))
    // trainModel.add(tf.layers.reshape({targetShape:[13,13,425]}))

    // trainModel.summary();

    // const newOutput = trainModel.apply(model.outputs[0]);
    // model = tf.model({inputs: model.inputs, outputs: model.outputs[0]}); 

    // const layer = model.getLayer('conv2d_9');
    // model = await tf.model({inputs: model.inputs, outputs: layer.output}); 
}
exports.app = async function(req, res) {
    if (!req.body) return res.sendStatus(400);
    res.render('index.html');
}
exports.tfTrain = async function(req,res) {
    console.log(dateNow(),'>>tfTrain()...');
    if (!req.body) return res.sendStatus(400);

    // const {} = req.body;

    model = await tf.loadLayersModel(URL);  
    model.summary();
     
    model.predict(tf.zeros([1, 416, 416, 3]));
    
    CLASSES = [];
    CLASSES.push('Class1');
    CLASSES.push('Class2');
    CLASSES.push('Class3');
    
    let epochs, optimizer, inputsAsTensor, targetTensor;

    // if (tf.getBackend()==='cpu') {
    //     tf.setBackend('gpu');
    // }

    // Freeze the layers of the pre-trained model except for the last few
    for (let i = 0; i < model.layers.length; i++) {
        model.layers[i].trainable = true;
    } 

    let trainingDataInputs  = [];
    let trainingDataOutputs = [];

    let bufferT, resizedT, imageT, imageFeatures;

    async function calculateImageFeatures(_image) {
        const imageSize = 416;
        
        bufferT  = tf.node.decodeImage(_image);           
        resizedT = tf.image.resizeNearestNeighbor(bufferT, [imageSize, imageSize]);
        imageT   = await resizedT.div(tf.scalar(255.0)).expandDims();

        return imageT;    
    }
    
    let img = [];
    img[0] = {class:0, xmin:75, ymin:70, xmax:110, ymax:83, name:'train416.jpg'}
    img[1] = {class:0, xmin:115, ymin:60, xmax:149, ymax:166, name:'train416.jpg'}  
    img[2] = {class:0, xmin:150, ymin:69, xmax:184, ymax:184, name:'train416.jpg'}
    img[3] = {class:1, xmin:187, ymin:88, xmax:219, ymax:187, name:'train416.jpg'}
    img[4] = {class:1, xmin:221, ymin:90, xmax:256, ymax:187, name:'train416.jpg'}
    img[5] = {class:2, xmin:257, ymin:81, xmax:289, ymax:170, name:'train416.jpg'}
    img[6] = {class:2, xmin:295, ymin:82, xmax:330, ymax:180, name:'train416.jpg'}
    for (i=7;i<80;i++) {
        img.push({class:i-4, xmin:0, ymin:0, xmax:416, ymax:416, name:'train416.jpg'});
    }

    for (let i of img) {
        const trainImage = fs.readFileSync('./app/images/'+i.name); 

        imageFeatures = await calculateImageFeatures(trainImage); 
        trainingDataInputs.push(imageFeatures);    
    }
    
    // console.log(trainingDataInputs);
    
    inputsAsTensor  = await tf.concat(trainingDataInputs);
    
    const ANCHORS = [0.573, 0.677, 1.87, 2.06, 3.34, 5.47, 7.88, 3.53, 9.77, 9.17]; 
    function _alogistic(e) {      
    
        // return Math.log(1/e-1)
        return Math.log(e)
    }

    // inputsAsTensor = tf.zeros([1,416,416,3]);  
    zerosTensor = tf.zeros([1,13,13,425]);  

    for (let c of img) {
        
        pixl = zerosTensor.dataSync();
        // console.log(pixl) 

        //Normalize in 0~1:
        const x1 = c.xmin/416;
        const y1 = c.ymin/416;
        const x2 = c.xmax/416;
        const y2 = c.ymax/416;

        // console.log(x1)
        // console.log(y1)
        // console.log(x2)
        // console.log(y2)

        //Normalize coordinates to scores example
        // x1 = 0.010362043045461178  => -1.097084879875183 
        // y1 = -0.013428867794573307 => -0.6710431575775146
        // w1 = 0.18198972940444946 => -0.40344923734664917
        // h1 = 0.5270078182220459  => 0.25029298663139343 

        bx = x1 + (x2-x1)/2
        by = y1 + (y2-y1)/2
        bw = x2 - x1
        bh = y2 - y1

        // console.log(bx)
        // console.log(by)
        // console.log(bw)
        // console.log(bh)

        let offset = 0;
        for (xg = 0; xg<13; xg++) {
        for (yg = 0; yg<13; yg++) {        
            for (p = 0; p<5; p++) {
            //The center object of the trained image
            if (xg===Math.round(bx*13)&&yg===Math.round(by*13)&&p===Math.round(bw*5/2+bh*5/2)) { 
                // Transfer to 0~1 corresponding to each grid cell:
                // pixl[offset++] = _alogistic(bx*13) - xg
                // pixl[offset++] = _alogistic(by*13) - yg
                pixl[offset++] = bx*13 - xg
                pixl[offset++] = by*13 - yg
                pixl[offset++] = Math.log(bw*13/ANCHORS[p * 2])
                pixl[offset++] = Math.log(bh*13/ANCHORS[p * 2 + 1])              
                pixl[offset++] = 1            
                pixl[offset+c.class] = 1
                
            } else {       
                offset += 5;
            }
            offset += 80;
            }
        }
        }

        tfpixl = tf.tensor(pixl, [1,13,13,425], 'float32');
        // console.log(await tfpixl.array())

        trainingDataOutputs.push(tfpixl);
    }
        
    targetTensor    = await tf.concat(trainingDataOutputs);  
    // console.log(targetTensor.dataSync())

    // model.summary();

    optimizer = tf.train.adam();
    model.compile({
        optimizer: optimizer,
        loss: 'binaryCrossentropy',
        // loss: 'categoricalCrossentropy',
        metrics: ['accuracy'],
    });

    // console.log(inputsAsTensor);
    // console.log(targetTensor); 
    
    epochs = 5;  
    await model.fit(inputsAsTensor, targetTensor, {    
        shuffle   : true, 
        batchSize : 32,     
        // validationSplitRetrain : 0.99,
        epochs    : epochs, 
        // callbacks : { onEpochEnd: async (epoch,logs) => {
        // progress.value = epoch/(epochs-1)*100;
        // console.log('Epoch', epoch, logs)
        // }}
    });

    inputsAsTensor.dispose();
    targetTensor.dispose();  

    // await model.save('./app/model/newmodel');   
    
    const predictImage = fs.readFileSync('./app/images/train.jpg'); 
    imageFeatures = await calculateImageFeatures(predictImage); 
    outputs = model.predict(imageFeatures);
    predictions = await outputs.array();

    try {        
        await res.send({status:'success', data: predictions});         
    } catch(err) {
        console.log(err);
    }
  
}