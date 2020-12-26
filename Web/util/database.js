const Sequelize = require('sequelize');

const sequelize = new Sequelize('three_lines', 'root', '12341234', {
  dialect: 'mysql',
  host: 'localhost'
});

module.exports = sequelize;