package com.rajdeep;

import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.classifiers.trees.RandomForest;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;
import weka.classifiers.Evaluation;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.data.category.DefaultCategoryDataset;

import javax.swing.*;
import java.io.File;
import java.util.Random;
import java.util.*;

public class RFExperiment {
    public static void main(String[] args) throws Exception {

        // Load dataset
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File("data/Machine Learning Datasets Updated/31. Random Forest/Salary_Experience.csv"));
        Instances data = loader.getDataSet();
        data.setClassIndex(data.numAttributes() - 1);

        ReplaceMissingValues rm = new ReplaceMissingValues();
        rm.setInputFormat(data);
        data = Filter.useFilter(data, rm);

        // Train-test split
        data.randomize(new Random(1));
        int trainSize = (int) (data.numInstances() * 0.8);
        Instances train = new Instances(data, 0, trainSize);
        Instances test = new Instances(data, trainSize, data.numInstances() - trainSize);

         DefaultCategoryDataset dataset = new DefaultCategoryDataset();
         double min = Integer.MAX_VALUE;
         int tre = 0;
         for( int i = 1;i<=200;i+=5)
         {
             RandomForest rf = new RandomForest();
             rf.setNumIterations(i);
             rf.buildClassifier(train);
             Evaluation eval = new Evaluation(train);
             eval.evaluateModel(rf, test);
             double rmse = eval.rootMeanSquaredError();
             if(rmse<min)
             {
                 min = rmse;
                 tre = i;
             }
             dataset.addValue(rmse , "RMSE" ,Integer.toString(i) );
         }
         JFreeChart chart = ChartFactory.createLineChart("Random Forest RMSE vs Number of Trees",
                 "Trees",
                 "RMSE",dataset
                 );
        JFrame frame = new JFrame("Random Forest Experiment");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.add(new ChartPanel(chart));
        frame.pack();
        frame.setVisible(true);


    }
}

