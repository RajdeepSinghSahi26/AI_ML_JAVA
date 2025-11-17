package com.rajdeep;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.classifiers.trees.J48;
import weka.filters.Filter;
import weka.classifiers.Evaluation;

import weka.filters.unsupervised.attribute.ReplaceMissingValues;

import java.io.File;
import java.util.Random;

public class DecisionTree {
    public static void main(String[] args) throws Exception {

        // Load CSV
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File("data/Machine Learning Datasets Updated/30. Decision tree/cricket1.csv"));
        Instances data = loader.getDataSet();
        data.setClassIndex(data.numAttributes() - 1);

        // Handle missing values
        ReplaceMissingValues rm = new ReplaceMissingValues();
        rm.setInputFormat(data);
        data = Filter.useFilter(data, rm);

        data.randomize(new Random(0));
        int trainSize = (int) (data.numInstances() * 0.8);
        int testSize = data.numInstances() - trainSize;
        Instances train = new Instances(data, 0, trainSize);
        Instances test = new Instances(data, trainSize, testSize);



        // Build J48 (C4.5) decision tree
        J48 tree = new J48();
        tree.setUnpruned(false);
        tree.setConfidenceFactor(0.4f);
        tree.setMinNumObj(1);
        tree.buildClassifier(train);

        // Print tree
        Evaluation eval = new Evaluation(train);
        eval.evaluateModel(tree, test);

        System.out.println(tree);                  // print tree
        System.out.println(eval.toSummaryString()); // accuracy etc.
        System.out.println(eval.toMatrixString());
    }
}
