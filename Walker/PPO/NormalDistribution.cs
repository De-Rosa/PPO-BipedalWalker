using System;

namespace Physics.Walker.PPO;

public class NormalDistribution
{
    // https://stackoverflow.com/questions/218060/random-gaussian-variables
    public static float BoxMullerTransform(float mean, float std, Random random)
    {
        float uniform1 = (float) (1 - random.NextDouble());
        float uniform2 = (float) (1 - random.NextDouble()); 
        float randStdNormal = MathF.Sqrt(-2f * MathF.Log(uniform1)) * MathF.Sin(2f * MathF.PI * uniform2);
        return mean + (std * randStdNormal);
    }

    // https://en.wikipedia.org/wiki/Normal_distribution
    public static float ProbabilityDensity(float mean, float std, float action)
    {
        float constant = 1f / (std * MathF.Sqrt(2f * MathF.PI));

        float exponent = (action - mean) / std;
        exponent *= exponent;
        exponent /= -2f;

        return constant * MathF.Exp(exponent);
    }
}