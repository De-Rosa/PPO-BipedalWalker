using System;

namespace NEA.Walker.PPO;

// Dense layer class, represents two matrices - a weight and bias matrix.
// Gradients are stored every time back propagation is performed.
public class DenseLayer : Layer
{
    private Matrix _weights;
    private Matrix _biases;

    private Matrix _derivativeLossWrtWeights; // derivative of the loss with respect to the weights
    private Matrix _derivativeLossWrtBiases; // derivative of the loss with respect to the biases
    private Matrix _meanGradientWeights;
    private Matrix _meanGradientBiases;
    private Matrix _varianceGradientWeights;
    private Matrix _varianceGradientBiases;

    private int _iteration;

    public DenseLayer(int inputSize, int outputSize)
    {
        _weights = Matrix.FromXavier(outputSize, inputSize);
        _biases = Matrix.FromZeroes(outputSize, 1);

        _iteration = 0;

        _meanGradientWeights = Matrix.FromZeroes(outputSize, inputSize);
        _meanGradientBiases = Matrix.FromZeroes(outputSize, 1);
        _varianceGradientWeights = Matrix.FromZeroes(outputSize, inputSize);
        _varianceGradientBiases = Matrix.FromZeroes(outputSize, 1);
        _derivativeLossWrtWeights = Matrix.FromZeroes(outputSize, inputSize);
        _derivativeLossWrtBiases = Matrix.FromZeroes(outputSize, 1);
    }

    // Returns the output size of the dense layer (or the height of the weights matrix).
    public int GetOutputSize()
    {
        return _weights.GetHeight();
    }
    
    // Loads the weight/biases from a string taken from a file.
    public void Load(string contents)
    {
        int weightIndicator = contents.IndexOf("W", StringComparison.Ordinal) + 2;
        int biasIndicator = contents.IndexOf("B", StringComparison.Ordinal) + 2;
        string weights = contents.Substring(weightIndicator, biasIndicator - weightIndicator - 3);
        string biases = contents.Substring(biasIndicator, contents.Length - biasIndicator);

        _weights = Matrix.Load(_weights, weights);
        _biases = Matrix.Load(_biases, biases);
    }

    // Converts the weights/biases to a string representation.
    public string Save()
    {
        string line = "";
        line += "W " + Matrix.Save(_weights);
        line += " B " + Matrix.Save(_biases);
        return line;
    }

    // Feeds a matrix forward through the dense layer.
    public override Matrix FeedForward(Matrix matrix)
    {
        return _weights * matrix + _biases;
    }

    // Feeds a matrix backward through the dense layer.
    // Function is sourced from the github link below.
    // https://github.com/b2developer/SpidermanPPO/blob/main/PPO/Assets/Scripts/NeuralNetwork2/Dense.cs
    public override Matrix FeedBack(Matrix matrix, Matrix gradient)
    {
        _derivativeLossWrtBiases += (Matrix.Flatten(gradient));
        _derivativeLossWrtWeights += gradient * Matrix.Transpose(matrix);
        return Matrix.Transpose(_weights) * gradient;
    }

    // Adam optimiser adjusts the weights and biases inside of the dense layer along the gradients calculated during
    // back propagation.
    // https://optimization.cbe.cornell.edu/index.php?title=Adam
    public void Adam()
    {
        _iteration += 1;
        
        // momentum
        // mean2 = beta1 * mean1 + (1 - beta1) * gradient
        _meanGradientWeights = ((1 - Hyperparameters.Beta1) * _derivativeLossWrtWeights) + (Hyperparameters.Beta1 * _meanGradientWeights);
        _meanGradientBiases = ((1 - Hyperparameters.Beta1) * _derivativeLossWrtBiases) + (Hyperparameters.Beta1 * _meanGradientBiases);
        
        // rms
        // variance2 = beta2 * variance1 + (1 - beta2) * gradient^2
        _varianceGradientWeights = (Hyperparameters.Beta2 * _varianceGradientWeights) + (1 - Hyperparameters.Beta2) * Matrix.HadamardProduct(_derivativeLossWrtWeights, _derivativeLossWrtWeights);
        _varianceGradientBiases = (Hyperparameters.Beta2 * _varianceGradientBiases) + (1 - Hyperparameters.Beta2) * Matrix.HadamardProduct(_derivativeLossWrtBiases, _derivativeLossWrtBiases);
        
        // bias correction
        // mean2 = mean1 / (1 - beta1^t)
        // variance2 = variance1 / (1 - beta2^t)
        var correctedMeanGradientWeights = (_meanGradientWeights / (float) (1 - Math.Pow(Hyperparameters.Beta1, _iteration)));
        var correctedMeanGradientBiases = (_meanGradientBiases / (float) (1 - Math.Pow(Hyperparameters.Beta1, _iteration)));
        var correctedVarianceGradientWeights = (_varianceGradientWeights / (float) (1 - Math.Pow(Hyperparameters.Beta2, _iteration)));
        var correctedVarianceGradientBiases = (_varianceGradientBiases / (float) (1 - Math.Pow(Hyperparameters.Beta2, _iteration)));

        _weights = (_weights - (Hyperparameters.Alpha * Matrix.HadamardDivision(correctedMeanGradientWeights, Matrix.SquareRoot(correctedVarianceGradientWeights) + Hyperparameters.AdamEpsilon)));
        _biases = (_biases - (Hyperparameters.Alpha * Matrix.HadamardDivision(correctedMeanGradientBiases, Matrix.SquareRoot(correctedVarianceGradientBiases) + Hyperparameters.AdamEpsilon)));
    }

    // Zeros the gradients for the weight and biase matrices.
    public void ZeroGradients()
    {
        _derivativeLossWrtWeights.Zero();
        _derivativeLossWrtBiases.Zero();
    }
}