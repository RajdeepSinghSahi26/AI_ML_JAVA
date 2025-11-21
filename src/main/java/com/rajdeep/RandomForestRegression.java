package com.rajdeep;

import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.classifiers.trees.RandomForest;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;
import weka.filters.Filter;
import weka.classifiers.Evaluation;

import java.io.File;
import java.util.Random;

public class RandomForestRegression {
    public static void main(String[] args) throws Exception {

        // Load CSV
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File("data/Machine Learning Datasets Updated/31. Random Forest/Salary_Experience.csv"));
        Instances data = loader.getDataSet();

        // Set last column (Salary) as target
        data.setClassIndex(data.numAttributes() - 1);

        // Handle missing values
        ReplaceMissingValues rm = new ReplaceMissingValues();
        rm.setInputFormat(data);
        data = Filter.useFilter(data, rm);

        // Train-test split (80/20)
        data.randomize(new Random(1));
        int trainSize = (int)(data.numInstances() * 0.8);
        Instances train = new Instances(data, 0, trainSize);
        Instances test = new Instances(data, trainSize, data.numInstances() - trainSize);

        // Random Forest
        RandomForest rf = new RandomForest();
        rf.setNumIterations(100);
        rf.buildClassifier(train);

        // Evaluate
        Evaluation eval = new Evaluation(train);
        eval.evaluateModel(rf, test);

        // Print regression metrics
        System.out.print(rf.toString() + "\n");
        System.out.println(eval.toSummaryString());
    }
}

