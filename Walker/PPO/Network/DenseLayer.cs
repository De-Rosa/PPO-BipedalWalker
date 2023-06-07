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
    
    // Adam paper states "Good default settings for the tested machine learning problems are alpha=0.001, beta1=0.9, beta2=0.999 and epsilon=1e−8f"
    private const float Alpha = 0.01f; // step size
    private const float Beta1 = 0.9f; // 1st-order exponential decay
    private const float Beta2 = 0.999f; // 2nd-order exponential decay
    private const float Epsilon = 1e-8f; // prevent zero division

    private Matrix _derivativeLossDerivativeWeights;
    private Matrix _derivativeLossDerivativeBiases;
    private Matrix _meanGradientWeights;
    private Matrix _meanGradientBiases;
    private Matrix _varianceGradientWeights;
    private Matrix _varianceGradientBiases;

    private int _iteration;
    
    public DenseLayer(int inputSize, int outputSize)
    {
        _weights = Matrix.FromRandom(outputSize, inputSize);
        _biases = Matrix.FromZeroes(outputSize, 1);

        _iteration = 0;
        
        _meanGradientWeights = Matrix.FromZeroes(outputSize, inputSize);
        _meanGradientBiases = Matrix.FromZeroes(outputSize, 1);
        _varianceGradientWeights = Matrix.FromZeroes(outputSize, inputSize);
        _varianceGradientBiases = Matrix.FromZeroes(outputSize, 1);
        _derivativeLossDerivativeWeights = Matrix.FromZeroes(outputSize, inputSize);
        _derivativeLossDerivativeBiases = Matrix.FromZeroes(outputSize, 1);
    }

    private DenseLayer() {}

    public override Layer Clone()
    {
        return new DenseLayer
        {
            _weights = _weights.Clone(),
            _biases = _biases.Clone(),
            _derivativeLossDerivativeWeights = _derivativeLossDerivativeWeights.Clone(),
            _derivativeLossDerivativeBiases = _derivativeLossDerivativeBiases.Clone(),
            _meanGradientWeights = _meanGradientWeights.Clone(),
            _meanGradientBiases = _meanGradientBiases.Clone(),
            _varianceGradientWeights = _varianceGradientWeights.Clone(),
            _varianceGradientBiases = _varianceGradientBiases.Clone(),
            _iteration = _iteration
        };
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
        _derivativeLossDerivativeBiases += (Matrix.Flatten(gradient));
        _derivativeLossDerivativeWeights += gradient * Matrix.Transpose(matrix);
        return Matrix.Transpose(_weights) * gradient;
    }

    // https://optimization.cbe.cornell.edu/index.php?title=Adam
    // https://towardsdatascience.com/how-to-implement-an-adam-optimizer-from-scratch-76e7b217f1cc
    public void Adam()
    {
        _iteration += 1;
        
        // momentum
        // mean2 = beta1 * mean1 + (1 - beta1) * gradient
        _meanGradientWeights = ((1 - Beta1) * _derivativeLossDerivativeWeights) + (Beta1 * _meanGradientWeights);
        _meanGradientBiases = ((1 - Beta1) * _derivativeLossDerivativeBiases) + (Beta1 * _meanGradientBiases);
        
        // rms
        // variance2 = beta2 * variance1 + (1 - beta2) * gradient^2
        _varianceGradientWeights = (Beta1 * _varianceGradientWeights) + (1 - Beta1) * Matrix.HadamardProduct(_derivativeLossDerivativeWeights, _derivativeLossDerivativeWeights);
        _varianceGradientBiases = (Beta2 * _varianceGradientBiases) + (1 - Beta2) * Matrix.HadamardProduct(_derivativeLossDerivativeBiases, _derivativeLossDerivativeBiases);
        
        // bias correction
        // mean2 = mean1 / (1 - beta1^t)
        // variance2 = variance1 / (1 - beta2^t)
        var correctedMeanGradientWeights = (_meanGradientWeights / (float) (1 - Math.Pow(Beta1, _iteration)));
        var correctedMeanGradientBiases = (_meanGradientBiases / (float) (1 - Math.Pow(Beta1, _iteration)));
        var correctedVarianceGradientWeights = (_varianceGradientWeights / (float) (1 - Math.Pow(Beta2, _iteration)));
        var correctedVarianceGradientBiases = (_varianceGradientBiases / (float) (1 - Math.Pow(Beta2, _iteration)));

        _weights = (_weights - (Alpha * Matrix.HadamardDivision(correctedMeanGradientWeights, Matrix.SquareRoot(correctedVarianceGradientWeights) + Epsilon)));
        _biases = (_biases - (Alpha * Matrix.HadamardDivision(correctedMeanGradientBiases, Matrix.SquareRoot(correctedVarianceGradientBiases) + Epsilon)));
    }

    public void ZeroGradients()
    {
        _derivativeLossDerivativeWeights.Zero();
        _derivativeLossDerivativeBiases.Zero();
    }
}