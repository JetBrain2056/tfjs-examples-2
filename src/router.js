const Router = require('express');
const router = new Router();
const controller = require('./controller.js');

//assuming app is express Object.
router.get('/', controller.app); 

router.get('/tftrain', controller.tfTrain);

module.exports = { router }   
