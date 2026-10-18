const fs   = require('fs');
// const path = require('path');
// const tf   = require("@tensorflow/tfjs");
const tf = require('@tensorflow/tfjs-node');
const { pid } = require('process');
//const tf = require('@tensorflow/tfjs-node-gpu');
// const URL   = 'http://localhost:9000/app/model/model.json';
const URL = 'https://raw.githubusercontent.com/MikeShi42/yolo-tiny-tfjs/master/model2.json';

let model;

function dateNow() {
    let date = new Date(Date.now() - (new Date()).getTimezoneOffset() * 60000);
    return date.toISOString().slice(0, 19).replace('T', ' ');
}

function createYoloModel(numClasses) {
    
    const model = tf.sequential();           

    // Первый путь Darknet
    model.add(tf.layers.conv2d({
        filters: 16, 
        kernelSize: 3, 
        strides: 1, 
        inputShape: [416, 416, 3], 
        padding: 'same', 
        activation: 'leakyrelu',
        alpha: 0.1
    }));
    model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2], padding: 'same' }));
    
    model.add(tf.layers.conv2d({
        filters: 32, 
        kernelSize: 3, 
        strides: 1, 
        padding: 'same', 
        activation: 'leakyrelu',
        alpha: 0.1
    }));
    model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2], padding: 'same' }));
    
    model.add(tf.layers.conv2d({
        filters: 64, 
        kernelSize: 3, 
        strides: 1, 
        padding: 'same', 
        activation: 'leakyrelu',
        alpha: 0.1
    }));
    model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2], padding: 'same' }));
    
    model.add(tf.layers.conv2d({
        filters: 128, 
        kernelSize: 3, 
        strides: 1, 
        padding: 'same', 
        activation: 'leakyrelu',
        alpha: 0.1
    }));
    model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2], padding: 'same' }));
    
    model.add(tf.layers.conv2d({
        filters: 256, 
        kernelSize: 3, 
        strides: 1, 
        padding: 'same', 
        activation: 'leakyrelu',
        alpha: 0.1
    }));
    model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2], padding: 'same' }));
    
    // Второй путь
    model.add(tf.layers.conv2d({
        filters: 512, 
        kernelSize: 3, 
        strides: 1, 
        padding: 'same', 
        activation: 'leakyrelu',
        alpha: 0.1
    }));
    model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [1, 1], padding: 'same' }));
    
    model.add(tf.layers.conv2d({
        filters: 1024, 
        kernelSize: 3, 
        strides: 1, 
        padding: 'same', 
        activation: 'leakyrelu',
        alpha: 0.1
    }));
    
    model.add(tf.layers.conv2d({
        filters: 1024, 
        kernelSize: 3, 
        strides: 1, 
        padding: 'same', 
        activation: 'leakyrelu',
        alpha: 0.1
    }));
    
    // Выходной слой
    const numAnchors = 5; // стандартное количество анкоров для YOLO-tiny
    const filters = numAnchors * (numClasses + 5);
    
    model.add(tf.layers.conv2d({
        filters: filters, 
        kernelSize: 1, 
        strides: 1, 
        padding: 'same', 
        activation: 'linear'
    }));

    return model;
}
        
async function calculateImageFeatures(_image) {
        const imageSize = 416;
        
        const bufferT  = tf.node.decodeImage(_image);           
        const resizedT = tf.image.resizeNearestNeighbor(bufferT, [imageSize, imageSize]);
        const imageT   = await resizedT.div(tf.scalar(255.0)).expandDims();

        return imageT;    
}

function makeTargetTensor(imgs, gridSize, numAnchors, numClasses) {
    const filters = numAnchors*(numClasses+5);
    const targets = [];

    for (const c of imgs) {
        const bx = (c.xmin + c.xmax) / 2 / 416;
        const by = (c.ymin + c.ymax) / 2 / 416;
        const bw = (c.xmax - c.xmin) / 416;
        const bh = (c.ymax - c.ymin) / 416;

        if (bx < 0 || by < 0 || bx >= gridSize || by >= gridSize) {
            continue; // пропустить некорректные координаты
        }
            
        const gx = Math.floor(bx * gridSize);
        const gy = Math.floor(by * gridSize);
        
        const data = new Float32Array(gridSize * gridSize * filters).fill(0);

        const baseIdx = (gy * gridSize +gx) * filters;
        const anchorsIdx = 0;
        const offset = anchorsIdx * (numClasses + 5);

        data[baseIdx + offset] = bx - gx / gridSize;
        data[baseIdx + offset + 1] = by - gy / gridSize;
        data[baseIdx + offset + 2] = Math.log(bw);
        data[baseIdx + offset + 3] = Math.log(bh);
        data[baseIdx + offset + 4] = 1; // confidence

        // Установка класса
        data[baseIdx + offset + 5 + c.class] = 1;

        const tensor = tf.tensor(data, [1, gridSize, gridSize,filters], 'float32');
        targets.push(tensor);
    }

    const concat = tf.concat(targets, 0);
    return concat;

}

exports.app = async function(req, res) {
    if (!req.body) return res.sendStatus(400);
    res.render('index.html');
}

exports.tfTrain = async function(req,res) {
    console.log(dateNow(),'>>tfTrain()...');

    if (!req.body) return res.sendStatus(400);

    //v2 yolo_tiny
    const numClasses = 3;
    const gridSize = 13;
    const numAnchors = 5;
    const ANCHORS = [0.573, 0.677, 1.87, 2.06, 3.34, 5.47, 7.88, 3.53, 9.77, 9.17]; 
    const filters = numAnchors * (numClasses + 5);

    let baseModel;
    try {
        baseModel = await tf.loadLayersModel(URL);
    } catch (e) {
        console.error(e.message);
        baseModel = createYoloModel(numClasses);
    }  
        
    baseModel.summary();
     
     //Freeze the layers of the pre-trained model except for the last few    
    baseModel.layers.forEach((layer , i) => {
        layer.trainable = i === baseModel.layers.length - 1;
    });      
    
    const lastLayerConfig = baseModel.getLayer(baseModel.layers.length - 1).getConfig();
    const newLastLayer = tf.layers.conv2d({
        name: lastLayerConfig.name,
        filters: filters,
        kernelSize: lastLayerConfig.kernelSize,
        activation: lastLayerConfig.activation,
        padding: lastLayerConfig.padding,
        strides: lastLayerConfig.strides,
        kernelRegularizer: lastLayerConfig.kernelRegularizer,
        kernelInitializer: lastLayerConfig.kernelInitializer
    });

    const transferOutput = baseModel.layers[baseModel.layers.length - 2].output;    
    const newOutputs = newLastLayer.apply(transferOutput);
    const newModel = tf.model({inputs: baseModel.inputs, outputs: newOutputs});
    
     // Компиляция модели
    newModel.compile({ loss: 'meanSquaredError', optimizer: 'adam' , metrics: ['accuracy'],});

    // Вывод структуры модели
    newModel.summary();
   
    //demo for 3 classes
    const imgList = [
        {class:0, xmin:72, ymin:69, xmax:110, ymax:183, name:'train416.jpg'},
        {class:0, xmin:115, ymin:60, xmax:149, ymax:166, name:'train416.jpg'},  
        {class:0, xmin:150, ymin:69, xmax:184, ymax:184, name:'train416.jpg'},
        {class:1, xmin:187, ymin:88, xmax:219, ymax:187, name:'train416.jpg'},
        {class:1, xmin:221, ymin:90, xmax:256, ymax:187, name:'train416.jpg'},
        {class:2, xmin:257, ymin:81, xmax:289, ymax:170, name:'train416.jpg'},
        {class:2, xmin:295, ymin:82, xmax:330, ymax:180, name:'train416.jpg'}
    ];

    const inputs = [];
    for (const i of imgList) {
        const buffer = fs.readFileSync('./app/images/'+i.name); 
        const imageTensor = await calculateImageFeatures(buffer); 
        inputs.push(imageTensor);    
    }
    
    // console.log(trainingDataInputs);
    
    const inputsAsTensor  = tf.concat(inputs, 0);
  
    const targetTensor = makeTargetTensor(imgList, gridSize,numAnchors,numClasses);   
    console.log(targetTensor);      

    newModel.summary();
    
    
    await newModel.fit(inputsAsTensor, targetTensor, {
        epochs: 50,
        batchSize: 4,  // увеличенный размер батча
        validationSplit: 0.2,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                console.log(`Эпоха ${epoch}: loss = ${logs.loss}`);
                if (logs.loss < 0.01) {
                    return true; // остановить обучение
                }
            }
        },

        optimizer: tf.train.adam({
            learningRate: 0.001,
            decay: 0.0005
        }),
    });

    inputsAsTensor.dispose();
    targetTensor.dispose();  

    // await model.save('./app/model/newmodel');   
    
    const predictImage = fs.readFileSync('./app/images/train416.jpg'); 
    imageFeatures = await calculateImageFeatures(predictImage); 
    outputs = newModel.predict(imageFeatures);
    predictions = await outputs.array();

    try {        
        await res.send({status:'success', data: predictions});         
    } catch(err) {
        console.log(err);
    }
  
}
