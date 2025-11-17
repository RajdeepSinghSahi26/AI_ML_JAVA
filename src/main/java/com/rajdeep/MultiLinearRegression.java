package com.rajdeep;

import java.io.File;
import java.util.Random;

import weka.classifiers.functions.LinearRegression;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

public class MultiLinearRegression {

    public static void main(String[] args) throws Exception {

        CSVLoader loader = new CSVLoader();
        loader.setSource(new File("data/Machine Learning Datasets Updated/20. Multiple Linear Regression/homeprices.csv"));
        Instances data = loader.getDataSet();
        data.setClassIndex(data.numAttributes()-1);

        ReplaceMissingValues imputer = new ReplaceMissingValues();
        imputer.setInputFormat(data);
        data = Filter.useFilter(data, imputer);

        data.randomize(new java.util.Random(1));
        int trainSize = (int) Math.round(data.numInstances()*0.8);
        Instances train = new Instances(data, 0, trainSize);
        Instances test  = new Instances(data, trainSize, data.numInstances()-trainSize);

        LinearRegression ridge = new LinearRegression();
        ridge.setRidge(1e-4); // tune this (1e-4, 1e-3, 1e-2, 1e-1); higher => stronger regularization
        ridge.buildClassifier(train);

        // evaluate quickly: MAE/RMSE
        double sumAbs=0,sumSq=0,mean=0;
        for (int i=0;i<test.numInstances();i++) mean += test.instance(i).classValue();
        mean /= test.numInstances();
        double sst=0;
        for (int i=0;i<test.numInstances();i++){
            double actual = test.instance(i).classValue();
            double pred = ridge.classifyInstance(test.instance(i));
            sumAbs += Math.abs(actual-pred);
            sumSq += Math.pow(actual-pred,2);
            sst += Math.pow(actual-mean,2);
        }
        double mae = sumAbs/test.numInstances();
        double rmse = Math.sqrt(sumSq/test.numInstances());
        double r2 = 1 - (sumSq / sst);
        System.out.println("Ridge (lambda=1e-4) -> MAE: " + mae + " RMSE: " + rmse + " R2: " + r2);
        System.out.println("Model: " + ridge);

    }



}

