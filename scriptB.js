const storageID = "covid-binary";
let points;
let normalisedFeature, normalisedLabel;
let trainingFeatureTensor, testingFeatureTensor, trainingLabelTensor, testingLabelTensor;
let model = null;

//-----TOGGLE VISOR FUNCTION
async function toggleVisor() {
  tfvis.visor().toggle();
}

//-----CREATE MODEL FUNCTION
function createModel() {
    model = tf.sequential();

    model.add(tf.layers.dense({
      units: 10,
      useBias: true,
      activation: 'sigmoid',
      inputDim: 2,
    }));
    model.add(tf.layers.dense({
      units: 10,
      activation: 'sigmoid',
      useBias: true,
    }));
    // Output layer:
    model.add(tf.layers.dense({
      units: 1,
      activation: 'sigmoid',
      useBias: true,
    }));

    const optimizer = tf.train.adam();
    model.compile({
      loss: 'binaryCrossentropy',
      optimizer,
      metrics: ['accuracy'],
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
    epochs: 100,
    shuffle: true,
    validationSplit: 0.2,
    callbacks: {
      //onBatchEnd,
      onEpochEnd,
      onEpochBegin: async function(){
        await plotPredictionHeatmap();
        const layer = model.getLayer(undefined,0);
        tfvis.show.layer({name:"Layer1"},layer);
      }
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
    y: record.deaths,
    class: record.whoregion !== 1 ? 0 : record.whoregion,
  }));
  points = await pointsDataset.toArray();

  //Even amount of data for the training/testing split
  if (points.length % 2 !== 0) {
    points.pop();
  }

  tf.util.shuffle(points);
  plotClasses(points, "Regions");

  //Extract 2 features and out label (binary)
  const featureValues = points.map(p => [p.x, p.y]);
  const featureTensor = tf.tensor2d(featureValues);

  const labelValues = points.map(p => p.class);
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

  //Train the model
  const result = await trainModel(model, trainingFeatureTensor, trainingLabelTensor);
  console.log(result);
  const trainingLoss = result.history.loss.pop();
  console.log(`Training set loss: ${trainingLoss}`);

  //Validation
  const validationLoss = result.history.val_loss.pop();
  console.log(`Validation set loss: ${validationLoss}`);

  await plotPredictionHeatmap();

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
function normalise (tensor, previousMin = null, previousMax = null) {
    const featureDimensions = tensor.shape.length > 1 && tensor.shape[1];
    if (featureDimensions && featureDimensions > 1) {
      // More than one feature
      // Split into separate tensors
      const features = tf.split(tensor, featureDimensions, 1);

      // Normalise and find min/max values for each feature
      const normalisedFeatures = features.map((featureTensor, i) =>
        normalise(featureTensor, previousMin ? previousMin[i] : null, previousMax ? previousMax[i] : null));

      // Prepare return values
      // In this case the min and max properties will be arrays, with one
      // value for each feature
      const returnTensor = tf.concat(normalisedFeatures.map(f => f.tensor), 1);
      const min = normalisedFeatures.map(f => f.min);
      const max = normalisedFeatures.map(f => f.max);
      return { tensor: returnTensor, min, max};
    }
    else {
      // Just one feature
      const min = previousMin || tensor.min();
      const max = previousMax || tensor.max();
      const normalisedTensor = tensor.sub(min).div(max.sub(min));
      return {
        tensor: normalisedTensor,
        min,
        max
      };
    }
  }

//-----DENORMALISE THE DATA
function denormalise(tensor, min, max) {
    const featureDimensions = tensor.shape.length > 1 && tensor.shape[1];
    if (featureDimensions && featureDimensions > 1) {
      // More than one feature
      // Split into separate tensors
      const features = tf.split(tensor, featureDimensions, 1);

      // Denormalise
      const denormalised = features.map((featureTensor, i) => denormalise(featureTensor, min[i], max[i]));

      const returnTensor = tf.concat(denormalised, 1);
      return returnTensor;
    }
    else {
      const denormalisedTensor = tensor.mul(max.sub(min)).add(min);
      return denormalisedTensor;
    }
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

    await plotPredictionHeatmap();
    //Change user interface (show model in visor)
    tfvis.show.modelSummary({name: "Model summary"}, model);
    const layer = model.getLayer(undefined, 0);
    tfvis.show.layer({name: "Layer 1"}, layer);

    //Update status to the info from saved model and enable button
    document.getElementById("model-status").innerHTML = `Trained (saved ${modelInfo.dateSaved})`;
    document.getElementById("test-button").removeAttribute("disabled");
    document.getElementById("predict-button").removeAttribute("disabled");
  }else {
    alert("Could not load: no saved model found");
  }
}

//-----PREDICT FUNCTION
async function predict () {
    const predictionInputOne = parseInt(document.getElementById("prediction-input-1").value);
    const predictionInputTwo = parseInt(document.getElementById("prediction-input-2").value);
    if (isNaN(predictionInputOne)) {
      alert("Please enter a valid number");
    }
    else if (isNaN(predictionInputTwo)) {
      alert("Please enter a valid number");
    }
    else {
      tf.tidy(() => {
        const inputTensor = tf.tensor2d([[predictionInputOne, predictionInputTwo]]);
        const normalisedInput = normalise(inputTensor, normalisedFeature.min, normalisedFeature.max);
        const normalisedOutputTensor = model.predict(normalisedInput.tensor);
        const outputTensor = denormalise(normalisedOutputTensor, normalisedLabel.min, normalisedLabel.max);
        const outputValue = outputTensor.dataSync()[0];
        document.getElementById("prediction-output").innerHTML = `The likelihood of the patient being in the American region is: ${(outputValue*100).toFixed(1)}%`;
      });
    }
  }

//-----PLOT CLASSES
async function plotClasses (pointsArray, classKey, size = 400, equalizeClassSizes = false) {
  // Add each class as a series
  const allSeries = {};
  pointsArray.forEach(p => {
    // Add each point to the series for the class it is in
    const seriesName = `${classKey}: ${p.class}`;
    let series = allSeries[seriesName];
    if (!series) {
      series = [];
      allSeries[seriesName] = series;
    }
    series.push(p);
  });

  if (equalizeClassSizes) {
    // Find smallest class
    let maxLength = null;
    Object.values(allSeries).forEach(series => {
      if (maxLength === null || series.length < maxLength && series.length >= 100) {
        maxLength = series.length;
      }
    });
    // Limit each class to number of elements of smallest class
    Object.keys(allSeries).forEach(keyName => {
      allSeries[keyName] = allSeries[keyName].slice(0, maxLength);
      if (allSeries[keyName].length < 100) {
        delete allSeries[keyName];
      }
    });
  }
  tfvis.render.scatterplot(
    {
      name: `Deaths vs Confirmed`,
      styles: { width: "100%" }
    },
    {
      values: Object.values(allSeries),
      series: Object.keys(allSeries),
    },
    {
      xLabel: "Confirmed",
      yLabel: "Deaths",
      height: size,
      width: size*1.5,
    }
  );
}

//-----PLOT HEATMAP
async function plotPredictionHeatmap (name = "Predicted class", size = 400) {
    const [ valuesPromise, xTicksPromise, yTicksPromise ] = tf.tidy(() => {
      const gridSize = 50;
      const predictionColumns = [];
      // Heatmap order is confusing: columns first (top to bottom) then rows (left to right)
      // We want to convert that to a standard cartesian plot so invert the y values
      for (let colIndex = 0; colIndex < gridSize; colIndex++) {
        // Loop for each column, starting from the left
        const colInputs = [];
        const x = colIndex / gridSize;
        for (let rowIndex = 0; rowIndex < gridSize; rowIndex++) {
          // Loop for each row, starting from the top
          const y = (gridSize - rowIndex) / gridSize;
          colInputs.push([x, y]);
        }

        const colPredictions = model.predict(tf.tensor2d(colInputs));
        predictionColumns.push(colPredictions);
      }
      const valuesTensor = tf.stack(predictionColumns);

      const normalisedLabelsTensor = tf.linspace(0, 1, gridSize);
      const xTicksTensor = denormalise(normalisedLabelsTensor,
        normalisedFeature.min[0], normalisedFeature.max[0]);
      const yTicksTensor = denormalise(normalisedLabelsTensor.reverse(),
        normalisedFeature.min[1], normalisedFeature.max[1]);

      return [ valuesTensor.array(), xTicksTensor.array(), yTicksTensor.array() ];
    });

    const values = await valuesPromise;
    const xTicks = await xTicksPromise;
    const xTickLabels = xTicks.map(l => (l/1000).toFixed(1));
    const yTicks = await yTicksPromise;
    const yTickLabels = yTicks.map(l => (l/1000).toFixed(0));
    const data = {
      values,
      xTickLabels,
      yTickLabels,
    };

    tfvis.render.heatmap({
      name: `${name} (local)`,
      tab: "Predictions"
    }, data, { height: size });
    tfvis.render.heatmap({
      name: `${name} (full domain)`,
      tab: "Predictions"
    }, data, { height: size, domain: [0, 1] });
}
