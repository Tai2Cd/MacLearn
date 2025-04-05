using Tensorflow;
using Tensorflow.Keras;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Losses;
using Tensorflow.Keras.Layers;
using Tensorflow.Keras.Utils;
using Tensorflow.NumPy;
using Tensorflow.Keras.Models;
using CsvHelper;
using CsvHelper.Configuration;
using System.Globalization;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Optimizers;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using MathNet.Numerics.Statistics;
namespace Program
{
    public class HousingList
    {
        public float? Longitude { get; set; }
        public float? Latitude { get; set; }
        public float? HousingMedianAge { get; set; }
        public float? TotalRooms { get; set; }
        public float? TotalBedrooms { get; set; }
        public float? Population { get; set; }
        public float? Households { get; set; }
        public float? MedianIncome { get; set; }
        public float? MedianHouseValue { get; set; }
        public string OceanProximity { get; set; }
    }
    class Program
    {
        static void Main(string[] args)
        {
            Program p = new Program();
            p.Run();
        }
        void Run()
        {
            // Load the dataset
            var records = LoadCsvData("E:\\Codes\\MacStudyExp\\Exp1\\LinearRegression\\housing1.csv");

            // sperate features & label (without string)
            var features = records.Select(x => new float[] { x.Longitude.GetValueOrDefault(0), x.Latitude.GetValueOrDefault(0), x.HousingMedianAge.GetValueOrDefault(0), x.TotalRooms.GetValueOrDefault(0), x.TotalBedrooms.GetValueOrDefault(0), x.Population.GetValueOrDefault(0), x.Households.GetValueOrDefault(0), x.MedianIncome.GetValueOrDefault(0) }).ToArray();
            var label = records.Select(x => x.MedianHouseValue).ToArray();

            // deal with string column
            var oceanProximity = records.Select(x => x.OceanProximity).ToArray();
            var oceanProximityDict = new Dictionary<string, int>();
            for (int i = 0; i < oceanProximity.Length; i++)
            {
                if (!oceanProximityDict.ContainsKey(oceanProximity[i]))
                {
                    oceanProximityDict.Add(oceanProximity[i], oceanProximityDict.Count);
                }
            }
            var oceanProximityEncoded = oceanProximity.Select(x => oceanProximityDict[x]).ToArray();
            //add encoded column to features
            var finalFeatures = features.Select((f, i) => f.Concat(new float[] { oceanProximityEncoded[i] })).ToArray();

            //create Tensorflow model
            var featureArr = new float[finalFeatures.Length, 9];
            for (int i = 0; i < finalFeatures.Length; i++)
            {
                var _records = records.ElementAt(i);
                featureArr[i, 0] = _records.Longitude.GetValueOrDefault(0);
                featureArr[i, 1] = _records.Latitude.GetValueOrDefault(0);
                featureArr[i, 2] = _records.HousingMedianAge.GetValueOrDefault(0);
                featureArr[i, 3] = _records.TotalRooms.GetValueOrDefault(0);
                featureArr[i, 4] = _records.TotalBedrooms.GetValueOrDefault(0);
                featureArr[i, 5] = _records.Population.GetValueOrDefault(0);
                featureArr[i, 6] = _records.Households.GetValueOrDefault(0);
                featureArr[i, 7] = _records.MedianIncome.GetValueOrDefault(0);
                featureArr[i, 8] = oceanProximityEncoded[i];
            }

            var featureNDArr = np.array(featureArr);
            var labelArray = label.Select(x => (float)x).ToArray();
            var labelNDArr = np.array(labelArray);

            //split dataset as 70% for taining
            var trainSize = (int)(featureArr.Length / 9 * 0.7);
            var testSize = featureArr.Length / 9 - trainSize;

            // get train & test data
            int featureCount = featureArr.Length / 9;
            var trainFeatures = new float[trainSize, 9];
            var trainLabels = new float[trainSize];
            var tempFeatArr = featureNDArr.numpy().ToArray<float>();
            var tempFeatLength = tempFeatArr.Length / 9;
            for (int i = 0; i < trainSize; i++)
            {
                for (int j = 0; j < 9; j++)
                {
                    trainFeatures[i, j] = tempFeatArr[i * 9 + j];
                }
            }

            //Array.Copy(featureNDArr.numpy().ToArray<float>(), 0, trainFeatures, 0, trainSize);
            Array.Copy(labelNDArr.numpy().ToArray<float>(), 0, trainLabels, 0, trainSize);
            // for (int i = 0; i < trainSize; i++)
            // {
            //     for (int j = 0; j < 9; j++)
            //     {
            //         trainFeatures[i, j] = featureNDArr[i, j]; // 逐个元素复制
            //     }
            // }

            var testFeatures = new float[testSize, 9];
            var testLabels = new float[testSize];
            for (int i = trainSize; i < featureCount; i++)
            {
                for (int j = 0; j < 9; j++)
                {
                    testFeatures[i - trainSize, j] = tempFeatArr[i * 9 + j];
                }
            }
            Array.Copy(labelNDArr.numpy().ToArray<float>(), trainSize, testLabels, 0, testSize);

            //var model = BuildModel(finalFeatures.FirstOrDefault()?.Count() ?? 0);

            //train model
            //model.fit(featureNDArr, labelNDArr, epochs: 100, batch_size: 32);
            //model.save("E:\\Codes\\MacStudyExp\\Exp1\\LinearRegression\\housingTrained");
            //predict
            //var trainPrediction = model.predict(tf.constant(trainFeatures));
            //var testPrediction = model.predict(tf.constant(testFeatures));

            #region Calcuate R2
            //transfer XY arr as matrix
            var trainXMatrix = Matrix<float>.Build.DenseOfArray(trainFeatures);
            var trainYMatrix = Matrix<float>.Build.Dense(trainLabels.Length, 1, (i, j) => trainLabels[i]);
            var testXMatrix = Matrix<float>.Build.DenseOfArray(testFeatures);
            var testYMatrix = Matrix<float>.Build.Dense(testLabels.Length, 1, (i, j) => testLabels[i]);
            //Check dimention of each matrix
            Console.WriteLine("X " + trainXMatrix.RowCount + " " + trainXMatrix.ColumnCount);
            Console.WriteLine("Y " + trainYMatrix.RowCount + " " + trainYMatrix.ColumnCount);
            //Scale the matrix
            var X = FeatureScale(trainXMatrix);
            var XTest = FeatureScale(testXMatrix);
            //add a column of 1 as 1st column to X
            X = AddColumnOfOnes(X);
            XTest = AddColumnOfOnes(XTest);
            //prepare data of GradientDescent
            //theta is a colomn vextor. Get 2 theta as t1 for 梯度下降 and t2 for 正规方程
            var t1 = Matrix<float>.Build.Dense(10, 1, (i, j) => 0.0f);
            var t2 = t1;
            //first, check dimention of these vectors
            Console.WriteLine("t1 " + t1.RowCount + " " + t1.ColumnCount);
            var alpha = 0.01f;
            var iterations = 2000;//Init iterations times
            //create a 1-dimentional matrix to store cost every time
            var costHistory = Matrix<float>.Build.Dense(iterations, 1, (i, j) => 0.0f);


            t1 = GradientDescent(t1, X, trainYMatrix, costHistory, alpha, iterations);
            t2 = NormalEquation(X, trainYMatrix);

            //Calculate R2
            // Console.WriteLine("R2 of GradientDescent in train: " + R_2(t1, X, trainYMatrix));
            // Console.WriteLine("R2 of GradientDescent in test: " + R_2(t1, XTest, testYMatrix));
            // Console.WriteLine("R2 of NormalEquation in train: " +R_2(t2, X, trainYMatrix));
            // Console.WriteLine("R2 of NormalEquation in test: " + R_2(t2, XTest, testYMatrix));

            //get t1,t2 value
            //Console.WriteLine(CostFun(t1, X, trainYMatrix));
            //Console.WriteLine(CostFun(t2, X, trainYMatrix));

            #endregion
            //correlation
            double[] correlation = CalcuateCorrelation(trainXMatrix, trainYMatrix);
            Console.WriteLine("Correlation: ");
            Console.WriteLine("Longitude: {0}\nLatitude: {1}\nHousingMedianAge: {2}\nTotalRooms: {3}\nTotalBedrooms: {4}\nPopulation: {5}\nHouseholds: {6}\nMedianIncome: {7}\nOceanProximity: {8}", correlation[0], correlation[1], correlation[2], correlation[3], correlation[4], correlation[5], correlation[6], correlation[7], correlation[8]);
        }
        static List<HousingList> LoadCsvData(string path)
        {
            var config = new CsvConfiguration(CultureInfo.InvariantCulture)
            {
                MissingFieldFound = null// ignore missing fields
            };

            using var reader = new StreamReader(path);
            using var csv = new CsvReader(reader, config);
            var record = csv.GetRecords<HousingList>().ToList();

            //deal with null as avg
            #region excludeNull
            var totalLongitude = record.Select(r => r.Longitude).ToList();
            var avgLongitude = totalLongitude.Where(b => b.HasValue).Average(b => b.Value);
            foreach (var item in record)
            {
                if (!item.Longitude.HasValue)
                {
                    item.Longitude = avgLongitude;
                }
            }
            var totalLatitude = record.Select(r => r.Latitude).ToList();
            var avgLatitude = totalLatitude.Where(b => b.HasValue).Average(b => b.Value);
            foreach (var item in record)
            {
                if (!item.Latitude.HasValue)
                {
                    item.Latitude = avgLatitude;
                }
            }
            var totalHousingMedianAge = record.Select(r => r.HousingMedianAge).ToList();
            var avgHousingMedianAge = totalHousingMedianAge.Where(b => b.HasValue).Average(b => b.Value);
            foreach (var item in record)
            {
                if (!item.HousingMedianAge.HasValue)
                {
                    item.HousingMedianAge = avgHousingMedianAge;
                }
            }
            var totalTotalRooms = record.Select(r => r.TotalRooms).ToList();
            var avgTotalRooms = totalTotalRooms.Where(b => b.HasValue).Average(b => b.Value);
            foreach (var item in record)
            {
                if (!item.TotalRooms.HasValue)
                {
                    item.TotalRooms = avgTotalRooms;
                }
            }
            var totalTotalBedrooms = record.Select(r => r.TotalBedrooms).ToList();
            var avgTotalBedrooms = totalTotalBedrooms.Where(b => b.HasValue).Average(b => b.Value);
            foreach (var item in record)
            {
                if (!item.TotalBedrooms.HasValue)
                {
                    item.TotalBedrooms = avgTotalBedrooms;
                }
            }
            var totalPopulation = record.Select(r => r.Population).ToList();
            var avgPopulation = totalPopulation.Where(b => b.HasValue).Average(b => b.Value);
            foreach (var item in record)
            {
                if (!item.Population.HasValue)
                {
                    item.Population = avgPopulation;
                }
            }
            var totalHouseholds = record.Select(r => r.Households).ToList();
            var avgHouseholds = totalHouseholds.Where(b => b.HasValue).Average(b => b.Value);
            foreach (var item in record)
            {
                if (!item.Households.HasValue)
                {
                    item.Households = avgHouseholds;
                }
            }
            var totalMedianIncome = record.Select(r => r.MedianIncome).ToList();
            var avgMedianIncome = totalMedianIncome.Where(b => b.HasValue).Average(b => b.Value);
            foreach (var item in record)
            {
                if (!item.MedianIncome.HasValue)
                {
                    item.MedianIncome = avgMedianIncome;
                }
            }
            #endregion
            return record;
        }
        static Sequential BuildModel(int featureCount)
        {
            var model = new Sequential(new Tensorflow.Keras.ArgsDefinition.SequentialArgs());

            //add input layers
            var inputLayerArgs = new InputLayerArgs
            {
                InputShape = new int[] { featureCount }
            };
            model.add(new InputLayer(inputLayerArgs));

            //add output layers
            model.add(new Dense(new DenseArgs() { Units = 1, KernelRegularizer = new L2(0.001f), Activation = new Tensorflow.Keras.Activations().Linear }));
            //compile model
            var optimizer = new Adam(learning_rate: 0.001f);
            model.compile(optimizer, new MeanSquaredError());
            return model;
        }

        #region MatrixMethods
        static Matrix<float> FeatureScale(Matrix<float> data)
        {
            //calculate standard deviation for every column
            var sigma = Matrix<float>.Build.DenseOfRowArrays(CalcuateStandardDeviation(data));
            //calculate average for every column
            var avg = Matrix<float>.Build.DenseOfRowArrays(CalcuateAvg(data));
            //scale data
            for (int r = 0; r < data.RowCount; r++)
            {
                for (int c = 0; c < data.ColumnCount; c++)
                {
                    data[r, c] = (data[r, c] - avg[0, c]) / sigma[0, c];
                }
            }
            return data;
        }
        static float[] CalcuateStandardDeviation(Matrix<float> data)
        {
            int columnCount = data.ColumnCount;
            var stdDeviation = new float[columnCount];
            for (int i = 0; i < columnCount; i++)
            {
                stdDeviation[i] = (float)Math.Sqrt(data.Column(i).Select(x => Math.Pow(x, 2)).Average());
            }
            return stdDeviation;
        }
        static float[] CalcuateAvg(Matrix<float> data)
        {
            int columnCount = data.ColumnCount;
            var avg = new float[columnCount];
            for (int i = 0; i < columnCount; i++)
            {
                avg[i] = data.Column(i).Average();
            }
            return avg;
        }
        static Matrix<float> AddColumnOfOnes(Matrix<float> data)
        {
            int rowCount = data.RowCount;
            int columnCount = data.ColumnCount;
            var newData = Matrix<float>.Build.Dense(rowCount, columnCount + 1);
            for (int i = 0; i < rowCount; i++)
            {
                newData[i, 0] = 1;
            }
            for (int i = 0; i < rowCount; i++)
            {
                for (int j = 0; j < columnCount; j++)
                {
                    newData[i, j + 1] = data[i, j];
                }
            }
            return newData;
        }
        #endregion

        #region RegressionParamMethods
        static float CostFun(Matrix<float> theta, Matrix<float> X, Matrix<float> y)
        {
            var m = X.RowCount;
            var h = X * theta;
            var error = h - y;
            var sqErr = Vector<float>.Build.Dense(error.ToColumnMajorArray().Select(x => x * x).ToArray());
            return sqErr.Sum() / (2 * m);

        }
        static Matrix<float> GradientDescent(Matrix<float> theta, Matrix<float> X, Matrix<float> y, Matrix<float> cost, float alpha, int iterations)
        {
            for (int i = 0; i < iterations; i++)
            {
                cost[i, 0] = CostFun(theta, X, y);
                theta = theta - (alpha / X.RowCount) * (X.Transpose() * (X * theta - y));
            }
            return theta;
        }
        static Matrix<float> NormalEquation(Matrix<float> X, Matrix<float> y)
        {
            var theta = (X.Transpose() * X).Inverse() * X.Transpose() * y;
            return theta;
        }
        static double R_2(Matrix<float> theta, Matrix<float> XTest, Matrix<float> yTest)
        {
            var yPredict = XTest * theta;
            var yMean = yTest.Column(0).Average();
            var sse = (yTest - yPredict).Column(0).Select(x => x * x).Sum();
            var ssr = (yTest - yMean).Column(0).Select(x => x * x).Sum();
            var sst = ssr + sse;
            var r_2 = 1 - (sse / sst);
            return r_2;
        }
        #endregion

        #region Correlation
        static double[] CalcuateCorrelation(Matrix<float> X,Matrix<float> y)
        {
            int featureCount = X.ColumnCount;
            double[] correlation = new double[featureCount];
            for (int i = 0; i < featureCount; i++)
            {
                //get current feature
                var featureColumn= X.Column(i).ToArray();
                var labelArr = y.Column(0).ToArray();
                //Calculate Pearson Correlation
                correlation[i] = Correlation.Pearson(featureColumn.Select(x => (double)x),labelArr.Select(x => (double)x));
            }
            return correlation;
        }

        #endregion
    }
}
