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

import javax.swing.JFrame;
import java.awt.*;
import java.awt.geom.Ellipse2D;
import java.io.File;
import java.io.IOException;

public class SimpleLine {
    public static void main(String[] args) throws IOException {
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File("data/Machine Learning Datasets Updated/13. Advanced Data Analysis using pandas/emp.csv"));
        Instances data = loader.getDataSet();

        XYSeries series1 = new XYSeries("Machine Learning Datasets");
        XYSeries series2 = new XYSeries("Machine Learning Datasets2 ");
        for (int i = 0; i < data.numInstances(); i++) {
            String gender = data.instance(i).stringValue(2);
            if(gender.equals("M"))
                series1.add(data.instance(i).value(3), data.instance(i).value(4));
            else if(gender.equals("F"))
                series2.add(data.instance(i).value(3), data.instance(i).value(4));

        }
        XYSeriesCollection dataset = new XYSeriesCollection();
        dataset.addSeries(series1);
        dataset.addSeries(series2);

        JFreeChart chart = ChartFactory.createXYLineChart("Age vs Salary",
                "Age",
                "Salary",
                dataset);
        XYPlot plot = chart.getXYPlot();
        plot.setBackgroundPaint(Color.white);
        plot.setRangeGridlinePaint(Color.lightGray);
        plot.setDomainGridlinePaint(Color.lightGray);

        XYLineAndShapeRenderer renderer = new XYLineAndShapeRenderer();
        renderer.setSeriesPaint(0, Color.red);
        renderer.setSeriesPaint(1, Color.green);
        renderer.setSeriesStroke(0, new BasicStroke(3.0f));
        renderer.setSeriesShapesVisible(0, true);
        renderer.setSeriesShapesVisible(1, true);


        plot.setRenderer(renderer);

        chart.getTitle().setFont(new Font("SansSerif", Font.BOLD, 18));
        chart.getLegend().setItemFont(new Font("SansSerif", Font.PLAIN, 14));

        ChartPanel chartPanel = new ChartPanel(chart);
        JFrame frame = new JFrame();
        frame.add(chartPanel);
        frame.setSize(800, 600);
        frame.setVisible(true);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    }
}
