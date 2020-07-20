const storageID = "covid-regression";
let points;
let normalisedFeature, normalisedLabel;
let trainingFeatureTensor, testingFeatureTensor, trainingLabelTensor, testingLabelTensor;
let model = null;

//-----TOGGLE VISOR FUNCTION
async function toggleVisor() {
  tfvis.visor().toggle();
}

//-----CREATE MODEL FUNCTION
function createModel(){
  model = tf.sequential();
  model.add(tf.layers.dense({
    inputDim: 1,
    units: 1,
    activation: 'linear',
    useBias: true,
  }));
  const optimizer = tf.train.adam();
  model.compile({
    optimizer,
    loss: 'meanSquaredError'
  });
  return model;
}

//-----TRAIN MODEL FUNCTION
async function trainModel(model, trainingFeatureTensor, trainingLabelTensor){
  const { onEpochEnd } = tfvis.show.fitCallbacks({
    name: "Training Performance" },
    ['loss']
  );
  return model.fit(trainingFeatureTensor, trainingLabelTensor, {
    epochs: 20,
    shuffle: true,
    validationSplit: 0.2,
    callbacks: {
      //onBatchEnd,
      onEpochEnd,
      //onEpochBegin: async function(){
      //  await plotPredictionLine();
      //  const layer = model.getLayer(undefined,0);
      //  tfvis.show.layer({name:"Layer1"},layer);
      //}
    }
  });
}

//-----RUN FUNCTION
async function run(){
  await tf.ready();

  //Gets the data to plot
  const covidDataset = tf.data.csv('http://127.0.0.1:8080/covid.csv');
  const pointsDataset = covidDataset.map(record => ({
    x: record.confirmed,
    y: record.deaths
  }));
  points = await pointsDataset.toArray();

  //Even amount of data for the training/testing split
  if (points.length % 2 !== 0) {
    points.pop();
  }

  tf.util.shuffle(points);
  plot(points);

  //Extract x and y and put them into tensors
  const featureValues = points.map(p => p.x);
  const featureTensor = tf.tensor2d(featureValues, [featureValues.length, 1]);

  const labelValues = points.map(p => p.y);
  const labelTensor = tf.tensor2d(labelValues, [labelValues.length, 1]);

  //Normalize the tensors
  normalisedFeature = normalise(featureTensor);
  normalisedLabel = normalise(labelTensor);

  //Memory usage
  featureTensor.dispose();
  labelTensor.dispose();

  //Split the data for training and testing
  var splitAmount = [(labelValues.length) * 0.8, (labelValues.length) * 0.2];
  [trainingFeatureTensor, testingFeatureTensor] = tf.split(normalisedFeature.tensor, splitAmount);
  [trainingLabelTensor, testingLabelTensor] = tf.split(normalisedLabel.tensor, splitAmount);

  //Update status and enable buttons
  document.getElementById("model-status").innerHTML = "No model trained";
  document.getElementById("train-button").removeAttribute("disabled");
  document.getElementById("load-button").removeAttribute("disabled");
}
run();

//-----TRAIN FUNCTION
async function train() {
  //Disable all buttons when training and update status
  ["train", "test", "load", "predict", "save"].forEach(id => {
    document.getElementById(`${id}-button`).setAttribute("disabled", "disabled");
  });
  document.getElementById("model-status").innerHTML = "Training...";

  //Create the model and optimize it
  model = createModel();
  model.summary();
  tfvis.show.modelSummary({name: "Model Summary", tab: 'Model'}, model);
  const layer = model.getLayer(undefined,0);
  tfvis.show.layer({name: "Layer 1", tab: 'Model Inspection'}, layer);

  await plotPredictionLine();

  //Train the model
  const result = await trainModel(model, trainingFeatureTensor, trainingLabelTensor);
  console.log(result);
  const trainingLoss = result.history.loss.pop();
  console.log(`Training set loss: ${trainingLoss}`);

  //Validation
  const validationLoss = result.history.val_loss.pop();
  console.log(`Validation set loss: ${validationLoss}`);

  //Enable buttons and update status
  document.getElementById("model-status").innerHTML = `Trained (unsaved)\nLoss: ${trainingLoss.toPrecision(5)}\nValidation loss: ${validationLoss.toPrecision(5)}`;
  document.getElementById("test-button").removeAttribute("disabled");
  document.getElementById("save-button").removeAttribute("disabled");
  document.getElementById("predict-button").removeAttribute("disabled");
}

//-----TEST FUNCTION
async function test() {
  const lossTensor = model.evaluate(testingFeatureTensor, testingLabelTensor);
  const loss = ( await lossTensor.dataSync())[0];
  console.log(`Testing Loss: ${loss}`);

  //Update status
  document.getElementById("testing-status").innerHTML = `Testing set loss: ${loss.toPrecision(5)}`;
}

//-----NORMALISE THE DATA
function normalise(tensor, previousMin = null, previousMax = null) {
  //if no previous min and max, calculate otherwise they have priority
  const max = previousMin || tensor.max();
  const min = previousMax || tensor.min();
  const normalisedTensor = tensor.sub(min).div(max.sub(min));
  return {
    tensor: normalisedTensor,
    min,
    max
  };
}

//-----DENORMALISE THE DATA
function denormalise(tensor, min, max) {
  const denormalisedTensor = tensor.mul(max.sub(min)).add(min);
  return denormalisedTensor;
}

//-----SAVE FUNCTION
async function save() {
  const saveResults = await model.save(`localstorage://${storageID}`);
  document.getElementById("model-status").innerHTML = `Trained (saved ${saveResults.modelArtifactsInfo.dateSaved})`;//getting metadata
}

//-----LOAD FUNCTION
async function load() {
  //Get from local storage
  const storageKey = `localstorage://${storageID}`;

  //List models saved
  const models = await tf.io.listModels();
  const modelInfo = models[storageKey];

  //If there is a model, get back its metadata
  if (modelInfo) {
    //Load the model
    model = await tf.loadLayersModel(storageKey);

    //Change user interface (show model in vosor)
    tfvis.show.modelSummary({name: "Model summary"}, model);
    const layer = model.getLayer(undefined, 0);
    tfvis.show.layer({name: "Layer 1"}, layer);

    await plotPredictionLine();

    //Update status to the info from saved model and enable button
    document.getElementById("model-status").innerHTML = `Trained (saved ${modelInfo.dateSaved})`;
    document.getElementById("test-button").removeAttribute("disabled");
    document.getElementById("predict-button").removeAttribute("disabled");
  }else {
    alert("Could not load: no saved model found");
  }
}

//-----PREDICT FUNCTION
async function predict() {
  const predictionInput = parseInt(document.getElementById("prediction-input").value);
  if (isNaN(predictionInput)) {
    alert("Please enter a valid number");
  } else {
    tf.tidy(() => {//memory management as dealing with tensors
      //1D tensor normalised
      const inputTensor = tf.tensor1d([predictionInput]);
            console.log(inputTensor);
      const normalisedInput = normalise(inputTensor, normalisedFeature.min, normalisedFeature.max);
      console.log(normalisedFeature.min);
      console.log(normalisedFeature.max);
      const normalisedOutputTensor = model.predict(normalisedInput.tensor);

      //Denormalise output
      const outputTensor = denormalise(normalisedOutputTensor, normalisedLabel.min, normalisedLabel.max);
      const outputValue = outputTensor.dataSync()[0];
      console.log(outputValue);
      const outputValueRounded = outputValue.toFixed(0);

      document.getElementById("prediction-output").innerHTML = `The predicted number of deaths is <br>`
        + `<span style="font-size: 2em">${outputValueRounded}</span>`;
    });
  }
}

//-----PLOT THE DATA
async function plot(pointsArray, featureName, predictedPointsArray = null) {

  const values = [pointsArray.slice(0, 1000)];
  //const series = ["original"];
  if (Array.isArray(predictedPointsArray)) {
    values.push(predictedPointsArray);
    //series.push("predicted");
  }

  tfvis.render.scatterplot(
    {name: `${featureName} vs Deaths cases`},
    {values: [points]},
    {
      xLabel: 'Confirmed',
      yLabel: 'Deaths',
    });
}

//-----PLOT PREDICTION LINE
async function plotPredictionLine(){
  const [xs, ys] = tf.tidy(() => {
    const normalisedXs = tf.linspace(0, 1, 100);
    const normalisedYs = model.predict(normalisedXs.reshape([100, 1]));

    const xs = denormalise(normalisedXs, normalisedFeature.min, normalisedFeature.max);
    const ys = denormalise(normalisedYs, normalisedLabel.min, normalisedLabel.max);

    return [ xs.dataSync(), ys.dataSync() ];
  });

  //Transform values in format expected by plot()
  const predictedPoints = Array.from(xs).map((val, index) => {
    return { x: val, y: ys[index] };
    });
    await plot(points, "Deaths", predictedPoints);
}

//PLOT PARAMETERS FUNCTION (to try different weights and biases in console)
async function plotParams(weight, bias) {
  model.getLayer(null, 0).setWeights([
    tf.tensor2d([[weight]]), // Kernel (input multiplier)
    tf.tensor1d([bias]), // Bias
  ]);
  await plotPredictionLine();
  const layer = model.getLayer(undefined, 0);
  tfvis.show.layer({ name: "Layer 1" }, layer);
}
