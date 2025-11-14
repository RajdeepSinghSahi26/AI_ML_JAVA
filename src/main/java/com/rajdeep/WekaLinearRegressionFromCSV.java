package com.rajdeep;

import org.knowm.xchart.XYChart;
import org.knowm.xchart.XYChartBuilder;
import weka.classifiers.functions.LinearRegression;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.filters.*;
import weka.filters.unsupervised.instance.RemovePercentage;
import java.io.File;
import java.util.ArrayList;

public class WekaLinearRegressionFromCSV {

    public static void main(String[] args) throws Exception {

        // Load CSV
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File("data/Machine Learning Datasets Updated/19. Simple Linear Regression/Salary_Data.csv"));
        Instances data = loader.getDataSet();

        // **Set class index**
        data.setClassIndex(data.numAttributes() - 1);

        // Train-test split
        // 80% train

        RemovePercentage removeTrain = new RemovePercentage();
        removeTrain.setPercentage(20);
        removeTrain.setInputFormat(data);
        Instances train = Filter.useFilter(data, removeTrain);

        // 20% test
        RemovePercentage removeTest = new RemovePercentage();
        removeTest.setPercentage(20);
        removeTest.setInvertSelection(true);
        removeTest.setInputFormat(data);
        Instances test = Filter.useFilter(data, removeTest);

        System.out.println("Train size: " + train.numInstances());
        System.out.println("Test size: " + test.numInstances());

        LinearRegression model = new LinearRegression();
        model.buildClassifier(train);

        train.setClassIndex(train.numAttributes() - 1);
        ArrayList<Double> pred= new ArrayList<>();
        for(int i=0;i<test.numInstances();i++) {
            double predicted = model.classifyInstance(test.instance(i));
            pred.add(predicted);
        }
        System.out.println("Predictions: " + pred);
        XYChart chart = new XYChartBuilder().width(800).height(600).title("salary_graph").build();
    }
}
