const http = require('http');
const bodyParser = require('body-parser');
const request = require('request-promise');
const UserInput = require('../models/userInput');
const Summary = require('../models/summary');

exports.getIndex = (req, res, next) => {
    res.render('index', {
        pageTitle: "세줄요약좀 해줘",
    })
};

exports.postInput = (req, res, next) => {
    const userInput = req.body.userInput;
    UserInput.create({
        USER_INPUT: userInput
    })
        .then(result => {
            console.log("저장성공/저장성공/저장성공/저장성공/저장성공/저장성공/저장성공/저장성공/저장성공/저장성공/저장성공/저장성공/저장성공/저장성공/저장성공/저장성공/")
            res.redirect('/summary');
        })
        .catch(err => {
            console.log(err);
        })
};

exports.getSummary = (req, res, next) => {
    UserInput.findOne({
        limit: 1, order: [['createdAT', 'DESC']],
    })
        .then(results => {
            const userInput = results['USER_INPUT'];
            const options = {
                uri: 'http://127.0.0.1:5000/summary',
                method: 'POST',
                body: { 'user_input': userInput },
                json: true
            };
            return request(options)
                .then(result => {
                    const summary = result['summary']
                    res.render('summary', {
                        pageTitle: "옛다 결과다",
                        summary: summary,
                        userInput: userInput
                    });
                })
        })
        .catch(err => {
            console.log(err);
        })
};