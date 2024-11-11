Materials used:
  https://github.com/ModelDepot/tfjs-yolo-tiny

  ![image](https://github.com/JetBrain2056/tfjs-examples-2/assets/108869575/84b01ce8-ada0-4971-b089-9a8b912f89b4)

  //ChatGPT version///////////////
    
    const newLastLayer = tf.layers.dense({
        units: 3, 
        activation: 'softmax'
    });
    model.remove(model.layers[model.layers.length - 1]); 
    model.add(newLastLayer);                        
    model.compile({
        optimizer: 'adam',
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });

    const trainData = prepareTrainingData();
    
    await model.fit(trainData.xs, trainData.labels, {
        batchSize: 64,
        epochs: 5,
        validationSplit: 0.2 
    });
