const Sequelize = require('sequelize');
const sequelize = require('../util/database');

const SummaryResult = sequelize.define('summaryResult', {
  id : {
    type: Sequelize.INTEGER,
    autoIncrement: true,
    allowNull: false,
    primaryKey: true
  },
  summaryResult: Sequelize.STRING
});

module.exports = SummaryResult;