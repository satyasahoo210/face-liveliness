const path = require('path')

module.exports = {
  entry: './public/static/js/script.ts',
  resolve: {
    extensions: ['.ts', '.js']
  },
  module: {
    rules: [
      { test: /\.ts$/, loader: 'ts-loader'}
    ]
  },
  output: {
    filename: 'script.js',
    path: path.resolve(__dirname, 'public', 'static', 'js')
  },
  mode: 'development'
}
