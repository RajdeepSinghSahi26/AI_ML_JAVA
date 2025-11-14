package com.rajdeep;

import weka.classifiers.functions.LinearRegression;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;
import weka.filters.unsupervised.attribute.InterquartileRange;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.Remove;

import java.io.File;

public class train_test_split {

    public static void main(String[] args) throws Exception {

        // 1) LOAD CSV
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File("data/Machine Learning Datasets Updated/20. Multiple Linear Regression/homeprices.csv"));
        Instances data = loader.getDataSet();

        // 2) Set target column
        data.setClassIndex(data.numAttributes() - 1);

        // 3) Fix missing values
        ReplaceMissingValues replaceMissing = new ReplaceMissingValues();
        replaceMissing.setInputFormat(data);
        data = Filter.useFilter(data, replaceMissing);

        // 4) Remove outliers using IQR
        InterquartileRange iqr = new InterquartileRange();
        iqr.setInputFormat(data);
        data = Filter.useFilter(data, iqr);

        // ---------- 5) NORMALIZE FEATURES ONLY ----------
        // Remove class column
        Remove removeClass = new Remove();
        removeClass.setAttributeIndicesArray(new int[]{data.classIndex()});
        removeClass.setInvertSelection(false);
        removeClass.setInputFormat(data);

        Instances featuresOnly = Filter.useFilter(data, removeClass);

        // Normalize features
        Normalize normalize = new Normalize();
        normalize.setInputFormat(featuresOnly);
        featuresOnly = Filter.useFilter(featuresOnly, normalize);

        // Extract class column only
        Remove keepClass = new Remove();
        keepClass.setAttributeIndicesArray(new int[]{data.classIndex()});
        keepClass.setInvertSelection(true);
        keepClass.setInputFormat(data);

        Instances targetOnly = Filter.useFilter(data, keepClass);

        // Merge features + target back
        Instances combined = Instances.mergeInstances(featuresOnly, targetOnly);
        combined.setClassIndex(combined.numAttributes() - 1);

        data = combined;
        // ------------------------------------------------

        // 6) SHUFFLE
        data.randomize(new java.util.Random(1));

        // 7) TRAIN-TEST SPLIT (80-20)
        int train_size = (int) Math.round(data.numInstances() * 0.8);
        int test_size = data.numInstances() - train_size;

        Instances train = new Instances(data, 0, train_size);
        Instances test = new Instances(data, train_size, test_size);

        // 8) TRAIN MODEL
        LinearRegression model = new LinearRegression();
        model.buildClassifier(train);

        // 9) EVALUATION
        double sumAbs = 0;
        double sumSq = 0;
        double ssTot = 0;

        double meanActual = 0;
        for (int i = 0; i < test.numInstances(); i++) {
            meanActual += test.instance(i).classValue();
        }
        meanActual /= test.numInstances();

        for (int i = 0; i < test.numInstances(); i++) {

            double actual = test.instance(i).classValue();
            double predicted = model.classifyInstance(test.instance(i));

            sumAbs += Math.abs(predicted - actual);
            sumSq += Math.pow(predicted - actual, 2);
            ssTot += Math.pow(actual - meanActual, 2);
        }

        double mae = sumAbs / test.numInstances();
        double rmse = Math.sqrt(sumSq / test.numInstances());
        double r2 = (ssTot == 0) ? Double.NaN : 1 - (sumSq / ssTot);

        // 10) RESULTS
        System.out.println("MAE: " + mae);
        System.out.println("RMSE: " + rmse);
        System.out.println("R²: " + r2);
    }
}

