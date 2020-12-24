const Sequelize = require('sequelize');
const sequelize = require('../util/database');

const UserInterest = sequelize.define('userInterest', {
  id : {
    type: Sequelize.INTEGER,
    autoIncrement: true,
    allowNull: false,
    primaryKey: true
  },
  USER_INPUT: Sequelize.STRING
});

module.exports = UserInterest;
