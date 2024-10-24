using System;
using System.ComponentModel;
using System.Security;

namespace NEA.Walker.PPO.Network;

// Activation layer class, represents an activation function which applies to the output of the previous
// layer.
public abstract class ActivationLayer : Layer
{
    // Feeds forward a matrix through the activation layer, applying the activation function to each value in the matrix.
    public override Matrix FeedForward(Matrix matrix)
    {
        return Matrix.PerformOperation(matrix, Activation);
    }

    // Feeds back a matrix through the activation layer, applying the derivative activation to each value in the matrix.
    public override Matrix FeedBack(Matrix matrix, Matrix gradient)
    {
        return Matrix.HadamardProduct(gradient, Matrix.PerformOperation(matrix, DerivativeActivation));
    }

    // Activation function
    protected abstract float Activation(float value);

    // Derivative activation function
    protected abstract float DerivativeActivation(float value);

}

// ReLU Layer class, child to activation layer - represents the ReLU activation function.
public class ReLULayer : ActivationLayer
{
    protected override float Activation(float value)
    {
        return MathF.Max(0, value);
    }
    
    protected override float DerivativeActivation(float value)
    {
        return value < 0 ? 0 : 1;
    }
}

// Leaky ReLU Layer class, child to activation layer - represents the Leaky ReLU activation function.
public class LeakyReLULayer : ActivationLayer
{
    private const float Alpha = 0.2f;

    protected override float Activation(float value)
    {
        return MathF.Max(Alpha * value, value);
    }
    
    protected override float DerivativeActivation(float value)
    {
        return value < 0 ? Alpha : 1;
    }
}

// TanH Layer class, child to activation layer - represents the TanH activation function.
public class TanhLayer : ActivationLayer
{
    protected override float Activation(float value)
    {
        return MathF.Tanh(value);
    }
    
    protected override float DerivativeActivation(float value)
    {
        return (1 - (MathF.Tanh(value) * MathF.Tanh(value)));
    }
}