package com.rajdeep;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class LoadData {
    public static void main(String[] args) throws Exception {
        // Load CSV or ARFF
        DataSource source = new DataSource("data/Machine Learning Datasets Updated/19. Simple Linear Regression/homeprices.csv");
        Instances data = source.getDataSet();

        // Set class index (the attribute to predict)
        if (data.classIndex() == -1)
            data.setClassIndex(data.numAttributes() - 1);

        System.out.println(data);
    }
}
