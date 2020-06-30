

/**-------------STEP 1
- Load the data and format it.
- The JSON file contains other data about each car, however we extract only the ones we need for now, which is MPG and HP.
*/
async function getData()
{
    const carsDataReq = await fetch('https://storage.googleapis.com/tfjs-tutorials/carsData.json');
    const carsData = await carsDataReq.json();
    const cleaned = carsData.map(car => ({
        mpg : car.Miles_per_Gallon,
        horsepower: car.Horsepower,
    })).filter(car => (car.mpg != null && car.horsepower != null));

    return cleaned;

}// end method getData()


/**-------------STEP 2
 * Plot the data to visualize it. Visualizing it can give us an idea if there is a pattern/structure/relationship that a model can learn.
 */
async function run()
{
    const data = await getData();
    const values = data.map(d => ({
        x: d.horsepower,
        y:d.mpg,
    }));

    tfvis.render.scatterplot(
        {name : 'Horsepower vs Miles Per Gallon'},
        {values},
        {
            xLabel:'Horsepower',
            yLabel:'MPG',
            height: 300
        }
    );
    
    //
    const model = createModel();

    tfvis.show.modelSummary({name:'Model Summary'}, model);

    const tensorData = convertToTensor(data);
    const {inputs, labels} = tensorData;

    await trainModel(model, inputs, labels);
    console.log("Done Training!");

    // Make some predictions using the model and compare them to the
    // original data
    testModel(model, data, tensorData);

}// end method run()

document.addEventListener('DOMContentLoaded', run);



/**-------------STEP 3
 * As the model takes 1 input and learns to predict 1 number, its one-to-one mapping
 */
function createModel()
{

    // This instantiates a tf.Model object. This model is sequential as its input flow straight to the output in a sequence. 
    const model = tf.sequential();

    // A single input layer. The dense layer multiples the inputs with a matrix(in this case 'units:1' , 1 weight for each input) of weights and then adds a bias, to the weighted sum.
    // The inputshape is 1 as we have the variable of horse power as the only input.
    model.add(tf.layers.dense({inputShape:[1], units :1, useBias:true}));// In dense layers the bias is added by default, therefore 'useBias:true' is not necessary
    
    //model.add(tf.layers.dense({units:100,activation:'sigmoid'}));// a hidden can be added for better results?
    
    // Output layer. As the hidden layer has only 1 unit, the output layer is not required for the outputs! However, defining a separate output layer allows us to modify
    // the number of units in the hidden layer while keeping the one-to-one mapping of the input and output.
    model.add(tf.layers.dense({units:1, activation:'sigmoid', useBias: true}));

    return model;
}

    //const model = await createModel();

    //tfvis.show.modelSummary({name:'Model Summary'}, model);





/**-------------STEP 4
 * Prepare the data for training. We will convert the input data into tensors, as well as the important practices of normalization and shuffling.
 */
 function convertToTensor(data)
 {
    return tf.tidy(() => {

        // Step 1 - Shuffle the data/ Randommize it for a variety of data in each batch.
        tf.util.shuffle(data);

        // Step 2 - Convert data to tensor.
        const inputs = data.map(d => d.horsepower);
        const labels = data.map(d => d.mpg);

        const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]); // (examples, [num of examples, num of features per example])
        const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

        // Step 3 - Normalize the data to the range 0 - 1 using min-max scaling. Normalization can be done before converting the data
        // to tensors, but better take advantage of vectorization.
        const inputMax = inputTensor.max();
        const inputMin = inputTensor.min();
        const labelMax = labelTensor.max();
        const labelMin = labelTensor.min();

        const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
        const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));

        return {
            inputs : normalizedInputs,
            labels : normalizedLabels,

            // Return the min/max bounds so we can use them later to un-normalize outputs and to normalize future data.

            inputMax,
            inputMin,
            labelMax,
            labelMin,
        }

    });
 }// end method convertToTensor()




 /**-------------STEP 5
  * Train the model. 
  */
  async function trainModel(model, inputs, labels)
  {
      model.compile({
        optimizer : tf.train.adam(),
        loss : tf.losses.meanSquaredError,
        metrics: ['mse'],
      });

      const batchSize = 32;
      const epochs = 100;

      return await model.fit(inputs, labels, {// 'model.fit' is the function called to start the training loop. This is an async function so we return the 
          batchSize,                          // promise it gives us so that the caller can determine when training is complete.
          epochs,
          shuffle: true,
          callbacks: tfvis.show.fitCallbacks( // 'tfvis.show.fitCallbacks' to generate charts for loss and mse
              {name: 'Training Performance'},
              ['loss', 'mse'],
              {height : 200, callbacks:['onEpochEnd']}
          )
      });
  }// end method trainModel()



   /**-------------STEP 6
  * Make predictions. 
  */
function testModel(model, inputData, normalizationData)
{
    const {inputMax, inputMin, labelMin, labelMax} = normalizationData;

    // Generate predictions for a uniform range of numbers between 0 and 1
    const [xs, preds] = tf.tidy(() => {

        const xs = tf.linspace(0,1,100); // 100 new examples generated
        const preds = model.predict(xs.reshape([100,1]));// 'model.predict()' used to feed the examples in the model

        // Data is un-normalized by the inverse of the min-max scaling
        const unNormXs = xs.mul(inputMax.sub(inputMin)).add(inputMin);
        const unNormPreds = preds.mul(labelMax.sub(labelMin)).add(labelMin);

        // return un-normalized data
        return [unNormXs.dataSync(), unNormPreds.dataSync()];// 'dataSync()' used to get a 'typedarray' of the values stored in a tensor, and process
                                                             // them in regular JS. This is a synchronous version of the '.data()' method which is generally preferred.
    });

    const predictedPoints = Array.from(xs).map((val, i) => {
        return {x : val, y: preds[i]}
    });

    const originalPoints = inputData.map(d => ({
        x : d.horsepower, y : d.mpg,
    }));

    tfvis.render.scatterplot(
        {name  : 'Model Predictions vs Original Data'},
        {values: [originalPoints, predictedPoints], series :['original', 'predicted']},
        {
            xLabel : 'Horsepower',
            yLabel : 'MPG',
            height : 300
        }
    );
}// end method testModel()

/**
 >>> Main takeaways
The steps in training a machine learning model include:

> Formulate your task:

- Is it a regression problem or a classification one?
- Can this be done with supervised learning or unsupervised learning?
- What is the shape of the input data? What should the output data look like?

> Prepare your data:

- Clean your data and manually inspect it for patterns when possible
- Shuffle your data before using it for training
- Normalize your data into a reasonable range for the neural network. Usually 0-1 or -1-1 are good ranges for numerical data.
- Convert your data into tensors

> Build and run your model:

- Define your model using 'tf.sequential' or 'tf.model' then add layers to it using 'tf.layers.*'
- Choose an optimizer (adam is usually a good one), and parameters like batch size and number of epochs.
- Choose an appropriate loss function for your problem, and an accuracy metric to help your evaluate progress. 'meanSquaredError' is a common loss function for regression problems.
- Monitor training to see whether the loss is going down

> Evaluate your model

- Choose an evaluation metric for your model that you can monitor while training. Once it's trained, try making some test predictions to get a sense of prediction quality.
 
*/