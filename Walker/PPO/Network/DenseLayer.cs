using System;
using System.Numerics;

namespace Physics.Walker.PPO;

// https://analyticsindiamag.com/a-complete-understanding-of-dense-layers-in-neural-networks/
// weight matrix * values in previous layer
// (output, input) = (m x n) matrix
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

    private DenseLayer()
    {
    }

    public override Layer Clone()
    {
        return new DenseLayer
        {
            _weights = _weights.Clone(),
            _biases = _biases.Clone(),
            _derivativeLossWrtWeights = _derivativeLossWrtWeights.Clone(),
            _derivativeLossWrtBiases = _derivativeLossWrtBiases.Clone(),
            _meanGradientWeights = _meanGradientWeights.Clone(),
            _meanGradientBiases = _meanGradientBiases.Clone(),
            _varianceGradientWeights = _varianceGradientWeights.Clone(),
            _varianceGradientBiases = _varianceGradientBiases.Clone(),
            _iteration = _iteration
        };
    }

    public void Load(string contents)
    {
        int weightIndicator = contents.IndexOf("W", StringComparison.Ordinal) + 2;
        int biasIndicator = contents.IndexOf("B", StringComparison.Ordinal) + 2;
        string weights = contents.Substring(weightIndicator, biasIndicator - weightIndicator - 3);
        string biases = contents.Substring(biasIndicator, contents.Length - biasIndicator);

        _weights = Matrix.Load(_weights, weights);
        _biases = Matrix.Load(_biases, biases);
    }

    public string Save()
    {
        string line = "";
        line += "W " + Matrix.Save(_weights);
        line += " B " + Matrix.Save(_biases);
        return line;
    }

    public Matrix GetWeights()
    {
        return _weights;
    }

    public override LayerType GetType()
    {
        return LayerType.DENSE;
    }

    public override Matrix FeedForward(Matrix matrix)
    {
        return _weights * matrix + _biases;
    }

    // https://github.com/b2developer/SpidermanPPO/blob/main/PPO/Assets/Scripts/NeuralNetwork2/Dense.cs
    // used function for feed back
    // adjust weights in the direction of the gradient
    public override Matrix FeedBack(Matrix matrix, Matrix gradient)
    {
        _derivativeLossWrtBiases += (Matrix.Flatten(gradient));
        _derivativeLossWrtWeights += gradient * Matrix.Transpose(matrix);
        return Matrix.Transpose(_weights) * gradient;
    }

    // https://optimization.cbe.cornell.edu/index.php?title=Adam
    // https://towardsdatascience.com/how-to-implement-an-adam-optimizer-from-scratch-76e7b217f1cc
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

    public void ZeroGradients()
    {
        _derivativeLossWrtWeights.Zero();
        _derivativeLossWrtBiases.Zero();
    }
}