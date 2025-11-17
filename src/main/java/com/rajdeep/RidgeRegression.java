package com.rajdeep;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.classifiers.functions.LinearRegression;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

import java.io.File;

public class RidgeRegression {

    public static void main(String[] args) throws Exception {

        // 1) LOAD CSV
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File("data/Machine Learning Datasets Updated/23. Ridge Regression/boston_houses.csv"));
        Instances data = loader.getDataSet();

        // Last column = target
        data.setClassIndex(data.numAttributes() - 1);

        // 2) HANDLE MISSING VALUES
        ReplaceMissingValues rm = new ReplaceMissingValues();
        rm.setInputFormat(data);
        data = Filter.useFilter(data, rm);
        data.randomize(new java.util.Random(42));

        // 3) TRAIN-TEST SPLIT (80-20)
        int total = data.numInstances();
        int train_size = (int) Math.round(total * 0.8);
        int test_size = total - train_size;

        Instances train = new Instances(data, 0, train_size);
        Instances test  = new Instances(data, train_size, test_size);

        // 4) TRAIN MODEL (RIDGE)
        LinearRegression model = new LinearRegression();
        model.setRidge(0.0001); // lambda = 1e-4
        model.buildClassifier(train);

        // 5) MANUAL METRICS: MAE, RMSE, R2
        double sumAbs = 0, sumSq = 0;

        double mean = 0;
        for (int i = 0; i < test.numInstances(); i++) {
            mean += test.instance(i).classValue();
        }
        mean /= test.numInstances();

        double sst = 0;
        for (int i = 0; i < test.numInstances(); i++) {

            double actual = test.instance(i).classValue();
            double pred   = model.classifyInstance(test.instance(i));

            sumAbs += Math.abs(actual - pred);
            sumSq  += Math.pow(actual - pred, 2);

            sst += Math.pow(actual - mean, 2);
        }

        double mae = sumAbs / test.numInstances();
        double rmse = Math.sqrt(sumSq / test.numInstances());
        double r2 = 1 - (sumSq / sst);

        System.out.println("Ridge (lambda=1e-4) -> MAE: " + mae +
                " RMSE: " + rmse +
                " R2: " + r2);

        System.out.println("\nModel: ");
        System.out.println(model);
    }
}
