// import modules
const path = require('path');
const express = require('express');
const bodyParser = require('body-parser');
const sequelize = require('./util/database');

// Setting App
const app = express();
app.set('view engine', 'ejs');
app.set('views', 'views');

// Importing routers
const  mainRoutes = require('./routes/main');

// Setting bodyparser, path, flash
app.use(bodyParser.urlencoded({ extended: false }));
app.use(bodyParser.json());
app.use(express.static(path.join(__dirname, 'public')));

// Routers
app.use(mainRoutes);

sequelize
.sync()
  .then(result => {
    app.listen(3000);
  })
  .catch(err => {
    console.log(err);
  });