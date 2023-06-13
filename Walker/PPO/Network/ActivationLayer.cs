using System;
using System.ComponentModel;
using System.Security;

namespace Physics.Walker.PPO.Network;

// https://binaryplanet.org/wp-content/uploads/2020/04/1-p_hyqAtyI8pbt2kEl6siOQ-1.png
public abstract class ActivationLayer : Layer
{
    public override Matrix FeedForward(Matrix matrix)
    {
        return Matrix.PerformOperation(matrix, Activation);
    }

    public override Matrix FeedBack(Matrix matrix, Matrix gradient)
    {
        return Matrix.HadamardProduct(gradient, Matrix.PerformOperation(matrix, DerivativeActivation));
    }

    public override LayerType GetType()
    {
        return LayerType.ACTIVATION;
    }

    protected abstract float Activation(float value);

    protected abstract float DerivativeActivation(float value);

    public abstract override Layer Clone();
}

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

    public override Layer Clone()
    {
        return new ReLULayer();
    }
}

public class LeakyReLULayer : ActivationLayer
{
    private const float Alpha = 0.01f;

    protected override float Activation(float value)
    {
        return MathF.Max(Alpha * value, value);
    }
    
    protected override float DerivativeActivation(float value)
    {
        return value < 0 ? Alpha : 1;
    }
    
    public override Layer Clone()
    {
        return new LeakyReLULayer();
    }
}

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
    
    public override Layer Clone()
    {
        return new TanhLayer();
    }
}