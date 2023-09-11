using System;

namespace Physics.Walker.PPO;

public class NormalDistribution
{
    // https://mathworld.wolfram.com/Box-MullerTransformation.html
    public static float BoxMullerTransform(float mean, float std, Random random)
    {
        float uniform1 = (float) random.NextDouble();
        float uniform2 = (float) random.NextDouble(); 
        if (uniform1 == 0) uniform1 = 1f; // avoid ln(0) = inf error
        float randStdNormal = MathF.Sqrt(-2f * MathF.Log(uniform1)) * MathF.Sin(2f * MathF.PI * uniform2);
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
