package com.rajdeep;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

import java.io.File;
import java.util.Random;

public class CrossValGridSearch {

    public static void main(String[] args) throws Exception {
        // === CONFIG ===
        String csvPath = "data/Machine Learning Datasets Updated/23. Ridge Regression/boston_houses.csv";
        int folds = 5;                     // common values: 5 or 10
        int seed = 42;                     // reproducible shuffling
        double[] ridgeCandidates = {0, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1}; // grid of lambda values
        // choose metric to minimize: here we use RMSE
        // === LOAD DATA ===
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File(csvPath));
        Instances data = loader.getDataSet();
        data.setClassIndex(data.numAttributes() - 1);

        // === PREPROCESS FILTERS ===
        // We'll use ReplaceMissingValues and Normalize. They must be in a FilteredClassifier so they run inside each CV fold.
        ReplaceMissingValues replaceMissing = new ReplaceMissingValues();
        Normalize normalize = new Normalize();

        // Chain filters: we will apply them by creating a small helper filter pipeline each time.
        // Note: FilteredClassifier accepts a single filter. If you need multiple, you can use weka.filters.MultiFilter,
        // but to keep it explicit here we will create a single MultiFilter.
        weka.filters.MultiFilter multiFilter = new weka.filters.MultiFilter();
        Filter[] filters = new Filter[] { replaceMissing, normalize };
        multiFilter.setFilters(filters);

        // === GRID SEARCH OVER ridgeCandidates ===
        double bestRidge = ridgeCandidates[0];
        double bestRmse = Double.POSITIVE_INFINITY;
        double bestMae = Double.POSITIVE_INFINITY;
        double bestR2 = Double.NEGATIVE_INFINITY;

        System.out.println("Starting grid search with " + folds + "-fold CV, seed=" + seed);

        for (double ridge : ridgeCandidates) {
            // build a FilteredClassifier that wraps LinearRegression
            LinearRegression lr = new LinearRegression();
            lr.setRidge(ridge);

            FilteredClassifier fc = new FilteredClassifier();
            fc.setClassifier(lr);
            fc.setFilter(multiFilter);

            // Perform cross-validated evaluation on the whole dataset
            Evaluation eval = new Evaluation(data);
            // randomize before CV for stratification of random sampling
            Instances randData = new Instances(data);
            randData.randomize(new Random(seed));

            // If class is nominal and you want stratified folds, use randData.stratify(folds)
            // For regression, stratify isn't used; randomize is enough.
            eval.crossValidateModel(fc, randData, folds, new Random(seed));

            double rmse = eval.rootMeanSquaredError();
            double mae = eval.meanAbsoluteError();
            double corr = eval.correlationCoefficient();
            double r2 = corr * corr;

            System.out.printf("ridge=%.8f  -> RMSE: %.4f  MAE: %.4f  R²: %.4f%n", ridge, rmse, mae, r2);

            if (rmse < bestRmse) {
                bestRmse = rmse;
                bestMae = mae;
                bestR2 = r2;
                bestRidge = ridge;
            }
        }

        System.out.println("\n=== BEST HYPERPARAMETER (by RMSE) ===");
        System.out.printf("Best ridge = %.8f  (RMSE=%.4f, MAE=%.4f, R²=%.4f)%n",
                bestRidge, bestRmse, bestMae, bestR2);

        // === Train final model on full training data with best ridge and evaluate on a held-out split ===
        // We'll do a final random 80/20 split to show final performance (optional)
        int total = data.numInstances();
        int trainSize = (int) Math.round(total * 0.8);
        int testSize = total - trainSize;

        // Randomize once more with same seed (so result is reproducible)
        Instances finalData = new Instances(data);
        finalData.randomize(new Random(seed));

        Instances train = new Instances(finalData, 0, trainSize);
        Instances test = new Instances(finalData, trainSize, testSize);

        // Build final FilteredClassifier with best ridge
        LinearRegression finalLR = new LinearRegression();
        finalLR.setRidge(bestRidge);

        FilteredClassifier finalFc = new FilteredClassifier();
        finalFc.setClassifier(finalLR);
        finalFc.setFilter(multiFilter);
        finalFc.buildClassifier(train);

        // Manual evaluation on held-out test
        Evaluation finalEval = new Evaluation(train);
        finalEval.evaluateModel(finalFc, test);

        System.out.println("\n=== FINAL EVALUATION ON 80/20 HOLDOUT ===");
        System.out.printf("MAE: %.4f%n", finalEval.meanAbsoluteError());
        System.out.printf("RMSE: %.4f%n", finalEval.rootMeanSquaredError());
        double corrFinal = finalEval.correlationCoefficient();
        System.out.printf("R²: %.4f%n", corrFinal * corrFinal);

        System.out.println("\nFinal model (coefficients):");
        // We can print the underlying linear model by getting it as string (FilteredClassifier includes filter info)
        System.out.println(finalLR); // note: printing finalLR before building may not show trained coeffs; finalFc.toString() also shows.
        System.out.println("\nFilteredClassifier summary:\n" + finalFc);

    }
}
