% test_nir_model.m
data = readtable('nir_test_data.csv');

% Extract spectra and true values
X_test = data{:, 1:100};
y_true = data{:, end-2:end};

% Load ONNX model
net = importONNXNetwork('nir_model.onnx');

% Predict
y_pred = predict(net, X_test);

% Display results
for i = 1:size(X_test, 1)
    fprintf('\nSample %d:\n', i);
    fprintf('True  -> Protein: %.2f, Fat: %.2f, Carbs: %.2f\n', ...
            y_true(i,1), y_true(i,2), y_true(i,3));
    fprintf('Pred  -> Protein: %.2f, Fat: %.2f, Carbs: %.2f\n', ...
            y_pred(i,1), y_pred(i,2), y_pred(i,3));
end

% Plot first sample
figure;
plot(linspace(800, 2500, 100), X_test(1,:));
xlabel('Wavelength (nm)');
ylabel('Absorbance');
title('Sample 1: NIR Spectrum');
grid on;