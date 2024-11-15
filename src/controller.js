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
        //chatGPT version////    

    function yoloLayers() {
        //const anchors = [1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52];
        //const numAnchors = anchors.length / 2;

        const model = tf.sequential();

        model.add(tf.layers.conv2d({ filters: 16, kernelSize: 3, strides: 1, inputShape: [416, 416, 3], padding: 'same', activation: 'LeakyReLU' }));
        model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2], padding: 'same' }));
        model.add(tf.layers.conv2d({ filters: 32, kernelSize: 3, strides: 1, padding: 'same', activation: 'LeakyReLU' }));
        model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2], padding: 'same' }));
        model.add(tf.layers.conv2d({ filters: 64, kernelSize: 3, strides: 1, padding: 'same', activation: 'LeakyReLU' }));
        model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2], padding: 'same' }));
        model.add(tf.layers.conv2d({ filters: 128, kernelSize: 3, strides: 1, padding: 'same', activation: 'LeakyReLU' }));
        model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2], padding: 'same' }));
        model.add(tf.layers.conv2d({ filters: 256, kernelSize: 3, strides: 1, padding: 'same', activation: 'LeakyReLU' }));
        model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2], padding: 'same' }));
        model.add(tf.layers.conv2d({ filters: 512, kernelSize: 3, strides: 1, padding: 'same', activation: 'LeakyReLU' }));
        model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [1, 1], padding: 'same' }));
        model.add(tf.layers.conv2d({ filters: 1024, kernelSize: 3, strides: 1, padding: 'same', activation: 'LeakyReLU' }));
        model.add(tf.layers.conv2d({ filters: 1024, kernelSize: 3, strides: 1, padding: 'same', activation: 'LeakyReLU' }));
        model.add(tf.layers.conv2d({ filters: 425, kernelSize: 1, strides: 1, padding: 'same' }));

        return model;
    }

    // Создание экземпляра модели
    model = yoloLayers();

    // Компиляция модели
    model.compile({ loss: 'binaryCrossentropy', optimizer: 'adam' , metrics: ['accuracy'],});

    // Вывод структуры модели
    model.summary();

    return model;
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
     
    //model.predict(tf.zeros([1, 416, 416, 3]));
    
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
        model.layers[i].trainable = false;
    } 

    model.layers[model.layers.length - 1].trainable = true; // Разморозка последнего слоя для дообучения

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
