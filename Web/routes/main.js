const path = require('path');
const express = require('express');
const mainController = require('../controllers/main');

const router = express.Router();

router.get('/', mainController.getIndex);

router.post('/get-user-input', mainController.postInput);

router.get('/summary', mainController.getSummary);

module.exports = router;