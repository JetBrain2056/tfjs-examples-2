const tf = require("@tensorflow/tfjs");
require('@tensorflow/tfjs-node');
require('@tensorflow/tfjs-node-gpu');
const URL   = 'http://localhost:9000/app/model/model.json';

let model;

function dateNow() {
    let date = new Date(Date.now() - (new Date()).getTimezoneOffset() * 60000);
    return date.toISOString().slice(0, 19).replace('T', ' ');
}
exports.app = async function(req, res) {
    if (!req.body) return res.sendStatus(400);
    model = await tf.loadLayersModel(URL);  
    model.summary();
    res.render('index.html');
}
exports.tfTrain = async function(req,res) {
    console.log(dateNow(),'>>tfTrain()...');
    if (!req.body) return res.sendStatus(400);

    // const {} = req.body;

     
    // if (tf.getBackend()==='cpu') {
    //   tf.setBackend('gpu');
    // }
     
    // console.log(tf.zeros([1, 416, 416, 3]));
    // console.log(tf.getBackend());

    // model.predict(tf.zeros([1, 416, 416, 3]));
    
    // let epochs = 5;  
    // await model.fit(inputsAsTensor, targetTensor, {    
    //   shuffle   : true, 
    //   batchSize : 32,     
    //   // validationSplitRetrain : 0.99,
    //   epochs    : epochs, 
    //   callbacks : { onEpochEnd: async (epoch,logs) => {
    //     progress.value = epoch/(epochs-1)*100;
    //     console.log('Epoch', epoch, logs)
    //   }}
    // });
  
    // inputsAsTensor.dispose();
    // targetTensor.dispose();  

    // try {        
    //     await res.send(data);         
    // } catch(err) {
    //     console.log(err);
    // }
  
}