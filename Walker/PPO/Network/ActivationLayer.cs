using System;
using System.ComponentModel;

namespace Physics.Walker.PPO;

public class ActivationLayer : Layer
{
    private static float ReLU(float value)
    {
        return MathF.Max(value, 0);
    }

    private static float ReLU(Matrix matrix)
    {
        throw new NotImplementedException();
    }
}