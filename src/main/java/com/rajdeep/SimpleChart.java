package com.rajdeep;

import com.opencsv.CSVReader;
import org.knowm.xchart.*;
import org.knowm.xchart.style.colors.XChartSeriesColors;
import org.knowm.xchart.style.lines.SeriesLines;
import org.knowm.xchart.style.markers.SeriesMarkers;

import java.io.FileReader;
import java.io.IOException;
import java.util.*;

public class SimpleChart {
    public static void main(String[] args) throws Exception {
        // Data — think of these as coordinates you’re spinning into view
        List<Double> xData = new ArrayList<>();
        List<Double> yData = new ArrayList<>();
        try (CSVReader csvReader = new CSVReader(new FileReader("data/Machine Learning Datasets Updated/19. Simple Linear Regression/Salary_Data.csv")))
        {
            String[] nextLine;
            csvReader.readNext();
            while ((nextLine = csvReader.readNext()) != null) {
                xData.add(Double.parseDouble(nextLine[0]));
                yData.add(Double.parseDouble(nextLine[1]));
            }
            // Create the chart frame
        }
        XYChart chart = new XYChartBuilder().width(1200).height(800).title("Salary_Data").xAxisTitle("X").yAxisTitle("Y").build();
        XYSeries series = chart.addSeries("Salary", xData, yData);
        series.setLineStyle(SeriesLines.NONE);
        series.setMarker(SeriesMarkers.DIAMOND);
        series.setShowInLegend(true);
        series.setLineWidth(1);
        series.setMarkerColor(XChartSeriesColors.BLUE);
        new SwingWrapper<>(chart).displayChart();
    }
}
