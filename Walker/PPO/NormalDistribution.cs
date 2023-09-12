using System;

namespace Physics.Walker.PPO;

public class NormalDistribution
{
    // https://mathworld.wolfram.com/Box-MullerTransformation.html
    public static float BoxMullerTransform(float mean, float std, Random random)
    {
        float float1 = (float) random.NextDouble();
        float float2 = (float) random.NextDouble(); 
        if (float1 == 0) float1 = 1f; // avoid ln(0) = inf error
        float randStdNormal = MathF.Sqrt(-2f * MathF.Log(float1)) * MathF.Sin(2f * MathF.PI * float2);
        return mean + (std * randStdNormal);
    }
    
    //https://ai.stackexchange.com/questions/40367/where-does-the-term-log-muu-mid-s-come-from
    public static float LogProbabilityDensity(float mean, float std, float action)
    {
        // -ln(std) - ln(sqrt(2pi)) - 0.5((x - mean) / std)^2
        float fraction = (action - mean) / std;
        fraction *= fraction;
        fraction /= -2;

        return -MathF.Log(std) - MathF.Log(MathF.Sqrt(2 * MathF.PI)) - fraction;
    }
}
