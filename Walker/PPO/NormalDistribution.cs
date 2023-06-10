using System;

namespace Physics.Walker.PPO;

public class NormalDistribution
{
    // https://stackoverflow.com/questions/218060/random-gaussian-variables
    public static float BoxMullerTransform(float mean, float std, Random random)
    {
        float uniform1 = (float) (1 - random.NextDouble());
        float uniform2 = (float) (1 - random.NextDouble()); 
        float randStdNormal = MathF.Sqrt(-2 * MathF.Log(uniform1)) * MathF.Sin(2 * MathF.PI * uniform2);
        return mean + (std * randStdNormal);
    }

    // https://en.wikipedia.org/wiki/Normal_distribution
    public static float ProbabilityDensity(float mean, float std, float action)
    {
        float constant = 1f / (std * MathF.Sqrt(2 * MathF.PI));

        float exponent = (action - mean) / std;
        exponent *= exponent;
        exponent /= -2f;

        return constant * MathF.Exp(exponent);
    }

    // https://stats.stackexchange.com/questions/404191/what-is-the-log-of-the-pdf-for-a-normal-distribution
    public static float LogProbabilityDensity(float mean, float std, float action)
    {
        float fraction = (action - mean);
        fraction *= fraction;
        fraction /= (2f * std * std);

        float lnPi = MathF.Log(MathF.Sqrt(2f * MathF.PI));

        return - fraction - MathF.Log(std) - lnPi;
    }
}