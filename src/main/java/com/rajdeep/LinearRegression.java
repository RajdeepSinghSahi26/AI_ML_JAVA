package com.rajdeep;
import com.opencsv.CSVReader;
import org.knowm.xchart.SwingWrapper;
import org.knowm.xchart.XYSeries;
import org.knowm.xchart.style.markers.SeriesMarkers;
import org.knowm.xchart.*;
import java.awt.*;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

public class LinearRegression{

    public static void main(String[] args) throws Exception {
        String filePath = "data/Machine Learning Datasets Updated/19. Simple Linear Regression/Salary_Data.csv";

        List<Double> xData = new ArrayList<>();
        List<Double> yData = new ArrayList<>();

        try (CSVReader reader = new CSVReader(new FileReader(filePath))) {
            String[] nextLine;
            reader.readNext(); // Skip header
            while ((nextLine = reader.readNext()) != null) {
                xData.add(Double.parseDouble(nextLine[0]));
                yData.add(Double.parseDouble(nextLine[1]));
            }
        }

        double meanX = mean(xData);
        double meanY = mean(yData);

        double numerator = 0.0;
        double denominator = 0.0;

        for (int i = 0; i < xData.size(); i++)
        {
            numerator += (xData.get(i) - meanX) * (yData.get(i) - meanY);
            denominator += Math.pow(xData.get(i) - meanX, 2);
        }

        double b = numerator / denominator; // slope
        double a = meanY - b * meanX;       // intercept

        System.out.println("Slope (b): " + b);
        System.out.println("Intercept (a): " + a);

        List<Double> yPred = new ArrayList<>();
        for (double x : xData) {
            yPred.add(a + b * x);
        }

        // === Plotting ===
        XYChart chart = new XYChartBuilder()
                .width(1200)
                .height(800)
                .title("Linear Regression — Salary Data")
                .xAxisTitle("Years of Experience")
                .yAxisTitle("Salary")
                .build();

        // Scatter points (actual data)
        XYSeries dataSeries = chart.addSeries("Actual Data", xData, yData);
        dataSeries.setMarker(SeriesMarkers.CIRCLE);
        dataSeries.setLineStyle(new BasicStroke(0f)); // no connecting lines

        // Regression line
        XYSeries regressionSeries = chart.addSeries("Regression Line", xData, yPred);
        regressionSeries.setMarker(SeriesMarkers.NONE);
        regressionSeries.setLineColor(java.awt.Color.RED);

        new SwingWrapper<>(chart).displayChart();

        System.out.println("Equation: y = " + a + " + " + b + "x");

    }

    private static double mean(List<Double> data) {
        double sum = 0.0;
        for (double val : data) sum += val;
        return sum / data.size();
    }
}
