package com.rajdeep;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import weka.core.Instances;
import weka.core.converters.CSVLoader;

import javax.swing.*;
import java.awt.*;
import java.io.File;
import java.io.IOException;

public class CHART_ANALYSIS {
    public static void main(String[] args) throws IOException {

        CSVLoader loader = new CSVLoader();
        loader.setSource(new File("data/Machine Learning Datasets Updated/20. Multiple Linear Regression/homeprices.csv"));
        Instances data = loader.getDataSet();

        XYSeries area = new XYSeries("Area vs Price");
        XYSeries beds = new XYSeries("Bedrooms vs Price");
        XYSeries age = new XYSeries("Age vs Price");

        for(int i = 0; i < data.numInstances(); i++) {
            area.add(data.instance(i).value(0), data.instance(i).value(3));
            beds.add(data.instance(i).value(1), data.instance(i).value(3));
            age.add(data.instance(i).value(2), data.instance(i).value(3));
        }

        XYSeriesCollection d1 = new XYSeriesCollection(area);
        XYSeriesCollection d2 = new XYSeriesCollection(beds);
        XYSeriesCollection d3 = new XYSeriesCollection(age);

        JFreeChart c1 = ChartFactory.createXYLineChart("Area", "Area", "Price", d1);
        XYPlot plot1 = c1.getXYPlot();
        XYLineAndShapeRenderer r1 = new XYLineAndShapeRenderer();

        r1.setSeriesPaint(0, Color.red);
        r1.setSeriesStroke(0, new BasicStroke(3.0f));
        r1.setSeriesShapesVisible(0, true);
        plot1.setRenderer(r1);

        JFreeChart c2 = ChartFactory.createXYLineChart("Bedrooms", "Bedrooms", "Price", d2);
        JFreeChart c3 = ChartFactory.createXYLineChart("Age", "Age", "Price", d3);

        ChartPanel p1 = new ChartPanel(c1);
        ChartPanel p2 = new ChartPanel(c2);
        ChartPanel p3 = new ChartPanel(c3);

        JPanel mainPanel = new JPanel(new GridLayout(1,3));
        mainPanel.add(p1);
        mainPanel.add(p2);
        mainPanel.add(p3);

        JFrame frame = new JFrame("Three Charts");
        frame.add(mainPanel);
        frame.setSize(1200, 400);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setVisible(true);

    }
}
