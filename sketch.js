// Neural Heads or Tails
// jshint esnext: true
//

const tsLength = [50,10];
const seqLen = 10;
const learningRate = 0.5;

var networks = [];
var traces = [];
var tries = 0;
var parameters = {
  previous: Array(seqLen).fill(0),
  fromLast: Array(2).fill(0),
};
var trainingSets = [[],[]];
var trainers = [];
var scores = Array(2).fill(0);
var humanDecision = 0;
var aiDecisions = [0,0];
var col = 255;

function setup() {
  createCanvas(400,400);
  networks.push(new synaptic.Architect.Perceptron(seqLen/2+4,13,23,1));
  networks.push(new synaptic.Architect.LSTM(1,20,20,1));
  trainers.push(new synaptic.Trainer(networks[0]));
  trainers.push(new synaptic.Trainer(networks[1]));
  traces.push({
    x:[0],
    y:[0],
    type: 'scatter',
    name: 'Perceptron',
  });
  traces.push({
    x:[0],
    y:[0],
    type: 'scatter',
    name: 'LSTM',
  });
  var data = traces.slice(0);
  Plotly.newPlot('plot', data);  
}

function runOnce(key) {
  if (key == 'Q' || key == 'A' || key == 'Z') {
    col = 255;
    var trainingPoints = [];
    
    var network = networks[0];    
    var previous = parameters.previous.slice(0);
    var inputs = previous.slice(0,5);
    var sum10 = previous.reduce((acc,cur) => acc + cur);
    inputs.push(sum10/seqLen);
    var sum5 = previous.reduce((acc,cur,i) => {if (i < seqLen/2) return acc + cur; else return acc;});
    inputs.push(sum5/(seqLen/2));
    inputs = inputs.concat(parameters.fromLast.map(x => x/seqLen));
    //console.log(inputs);
    outputs = network.activate(inputs);
    trainingPoints[0] = {
      input: inputs,
      output: [0],
    };
    if (outputs[0] > 0.5) {
      aiDecisions[0] = 1;
    } else {
      aiDecisions[0] = 0;
    }

    var network = networks[1];
    var inputs = parameters.previous.slice(0,1);
    outputs = network.activate(inputs);
    trainingPoints[1] = {
      input: inputs,
      output: [0],
    };
    if (outputs[0] > 0.5) {
      aiDecisions[1] = 1;
    } else {
      aiDecisions[1] = 0;
    }
    
    switch (key) {
      case 'Q':
        humanDecision = 0;
        break;
      case 'A':
        humanDecision = 1;
        break;
      case 'Z':
        humanDecision = round(random());
        break;
    }
    trainingPoints[0].output = [humanDecision];
    trainingPoints[1].output = [humanDecision];
    trainingSets[0].push(trainingPoints[0]);
    trainingSets[1].push(trainingPoints[1]);
    for (var i = 0; i < 2; i++) {
      if (trainingSets[i].length > tsLength[i]) {
        while (trainingSets[i].length > tsLength[i]) {
          trainingSets[i].shift();
        }
      }
    }
    parameters.previous.unshift(humanDecision);
    if (parameters.previous.length > seqLen) {
      while (parameters.previous.length > seqLen) {
        parameters.previous.pop();
      }
    }
    parameters.fromLast[humanDecision] = 0;
    parameters.fromLast[(humanDecision+1) % 2]++;
    
    if (aiDecisions[0] == humanDecision) {
      scores[0]++;
    } else {
      scores[0]--;
    }
    if (aiDecisions[1] == humanDecision) {
      scores[1]++;
    } else {
      scores[1]--;
    }
    
    tries++;
    traces[0].x.push(tries);
    traces[0].y.push(scores[0]);
    traces[1].x.push(tries);
    traces[1].y.push(scores[1]);
    
    var data = traces.slice(0);
    Plotly.newPlot('plot', data);

//     networks[0].propagate(learningRate,[humanDecision]); //will use trainingSet
//     networks[1].propagate(learningRate,[humanDecision]);
  }  
}

function keyPressed() {
  if (key != 'Z') {
    runOnce(key);
  }
}

function draw() {
  background(51);
  noStroke();
  col = lerp(col,51,0.05);
  fill(col);
  rectMode(CENTER);
  if (humanDecision === 0) {
    ellipse(width/2,height/4,width/4);
  } else {
    rect(width/2,height/4,width/4,width/4);
  }
  if (aiDecisions[0] === 0) {
    ellipse(width/4,3*height/4,width/4);
  } else {
    rect(width/4,3*height/4,width/4,width/4);
  }
  if (aiDecisions[1] === 0) {
    ellipse(3*width/4,3*height/4,width/4);
  } else {
    rect(3*width/4,3*height/4,width/4,width/4);
  }
  
  trainers[0].train(trainingSets[0],{
    rate: 0.1,
    iterations: 20,
    error: 0.9,
    shuffle: true,
    log: 0,
    cost: synaptic.Trainer.cost.CROSS_ENTROPY,
  });
  trainers[1].train(trainingSets[1],{
    rate: 0.1,
    iterations: 20,
    error: 0.7,
    shuffle: false,
    log: 0,
    cost: synaptic.Trainer.cost.CROSS_ENTROPY,
  });
  
  if (keyIsDown(90)) {
    runOnce('Z');
  }
  
  fill(255);
  text(frameRate(),10,10);
}
