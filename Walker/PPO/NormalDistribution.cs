using System;

namespace Physics.Walker.PPO;

public class NormalDistribution
{
    // https://stackoverflow.com/questions/218060/random-gaussian-variables
    public static float BoxMullerTransform(float mean, float std, Random random)
    {
        float uniform1 = (float) random.NextDouble();
        float uniform2 = (float) random.NextDouble(); 
        if (uniform1 == 0) uniform1 = 1f; // avoid ln(0) = inf error
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
    
    public static float LogProbabilityDensity(float mean, float std, float action)
    {
        float numerator = action - mean;
        float fraction = -(numerator * numerator) / (2 * std * std);
        float density = fraction - MathF.Log(std) - MathF.Log(MathF.Sqrt(2 * MathF.PI));

        return density;
    }
}
