package com.rajdeep;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class HousePriceModel {

    // ======================
    // Load CSV
    // ======================
    public static double[][] loadCSV(String path) throws Exception {
        List<double[]> rows = new ArrayList<>();
        BufferedReader br = new BufferedReader(new FileReader(path));
        br.readLine();
        String line;
        while ((line = br.readLine()) != null) {
            String[] parts = line.split(",");
            double[] row = new double[parts.length];

            for (int i = 0; i < parts.length; i++) {
                if (parts[i].trim().isEmpty()) {
                    row[i] = Double.NaN;   // mark missing
                } else {
                    row[i] = Double.parseDouble(parts[i]);
                }
            }
            rows.add(row);
        }
        br.close();
        return rows.toArray(new double[0][]);
    }

    // ======================
    // Fill Missing Values (Mean)
    // ======================
    public static void fillMissing(double[][] data) {
        int cols = data[0].length;

        for (int c = 0; c < cols; c++) {
            double sum = 0, count = 0;

            for (double[] row : data) {
                if (!Double.isNaN(row[c])) {
                    sum += row[c];
                    count++;
                }
            }

            double mean = sum / count;

            for (double[] row : data) {
                if (Double.isNaN(row[c])) {
                    row[c] = mean;
                }
            }
        }
    }

    // ======================
    // Extract X, y
    // ======================
    public static double[][] getX(double[][] data, int targetIndex) {
        int rows = data.length;
        int cols = data[0].length - 1;

        double[][] X = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            int k = 0;
            for (int j = 0; j < data[i].length; j++) {
                if (j != targetIndex) {
                    X[i][k++] = data[i][j];
                }
            }
        }
        return X;
    }

    public static double[] getY(double[][] data, int targetIndex) {
        double[] y = new double[data.length];
        for (int i = 0; i < data.length; i++) {
            y[i] = data[i][targetIndex];
        }
        return y;
    }

    // ======================
    // Fit Basic Linear Regression
    // ======================
    public static double[] fitLinear(double[][] X, double[] y) {
        int n = X.length;
        int d = X[0].length;

        double[][] XtX = new double[d][d];
        double[] Xty = new double[d];

        for (int i = 0; i < n; i++) {
            for (int a = 0; a < d; a++) {
                Xty[a] += X[i][a] * y[i];
                for (int b = 0; b < d; b++) {
                    XtX[a][b] += X[i][a] * X[i][b];
                }
            }
        }

        return gaussianSolve(XtX, Xty);
    }

    // ======================
    // Gaussian Solver
    // ======================
    public static double[] gaussianSolve(double[][] A, double[] b) {
        int n = b.length;

        for (int p = 0; p < n; p++) {
            double max = Math.abs(A[p][p]);
            int pivot = p;

            for (int i = p + 1; i < n; i++) {
                if (Math.abs(A[i][p]) > max) {
                    max = Math.abs(A[i][p]);
                    pivot = i;
                }
            }

            double[] temp = A[p];
            A[p] = A[pivot];
            A[pivot] = temp;

            double t = b[p];
            b[p] = b[pivot];
            b[pivot] = t;

            for (int i = p + 1; i < n; i++) {
                double alpha = A[i][p] / A[p][p];
                b[i] -= alpha * b[p];

                for (int j = p; j < n; j++) {
                    A[i][j] -= alpha * A[p][j];
                }
            }
        }

        double[] x = new double[n];
        for (int i = n - 1; i >= 0; i--) {
            double sum = b[i];
            for (int j = i + 1; j < n; j++) {
                sum -= A[i][j] * x[j];
            }
            x[i] = sum / A[i][i];
        }

        return x;
    }

    // ======================
    // Predict
    // ======================
    public static double[] predict(double[][] X, double[] coef) {
        double[] preds = new double[X.length];

        for (int i = 0; i < X.length; i++) {
            double s = 0;
            for (int j = 0; j < coef.length; j++) {
                s += coef[j] * X[i][j];
            }
            preds[i] = s;
        }
        return preds;
    }

    // ======================
    // Metrics
    // ======================
    public static void evaluate(double[] y, double[] pred) {
        double mae = 0, mse = 0;

        for (int i = 0; i < y.length; i++) {
            mae += Math.abs(y[i] - pred[i]);
            mse += Math.pow(y[i] - pred[i], 2);
        }

        mae /= y.length;
        mse /= y.length;

        double rmse = Math.sqrt(mse);

        System.out.println("MAE: " + mae);
        System.out.println("RMSE: " + rmse);
    }

    // ======================
    // MAIN
    // ======================
    public static void main(String[] args) throws Exception {
        double[][] data = loadCSV("data/Machine Learning Datasets Updated/20. Multiple Linear Regression/homeprices.csv");

        fillMissing(data);

        int target = 3;

        double[][] X = getX(data, target);
        double[] y = getY(data, target);

        double[] coef = fitLinear(X, y);
        System.out.println("Coefficients: " + Arrays.toString(coef));

        double[] preds = predict(X, coef);
        evaluate(y, preds);
    }
}

