using System;

namespace NEA.Walker.PPO;

// Normal distribution class, contains functions involving normal distribution mathematics.
// Used for action sampling and log probability calculation.
public class NormalDistribution
{
    // The Box-Muller Transform transforms a uniform distribution into a normal distribution with mean 0 and std 1.
    // We use this to sample a random number from a given normal distribution with mean 'mean' and std 'std'.
    // https://mathworld.wolfram.com/Box-MullerTransformation.html
    public static float BoxMullerTransform(float mean, float std, Random random)
    {
        float float1 = (float) random.NextDouble();
        float float2 = (float) random.NextDouble(); 
        if (float1 == 0) float1 = 1f; // avoid ln(0) = inf error
        float standardNormal = MathF.Sqrt(-2f * MathF.Log(float1)) * MathF.Sin(2f * MathF.PI * float2);
        return mean + (std * standardNormal);
    }
    
    // This is the log version of the PDF for normal distributions.
    // Used in the calculations for the PPO surrogate objective.
    //https://ai.stackexchange.com/questions/40367/where-does-the-term-log-muu-mid-s-come-from
    public static float LogProbabilityDensity(float mean, float std, float action)
    {
        // -ln(std) - ln(sqrt(2pi)) - 0.5((x - mean) / std)^2
        float fraction = (action - mean) / std;
        fraction *= fraction;
        fraction /= 2;

        return -MathF.Log(std) - MathF.Log(MathF.Sqrt(2 * MathF.PI)) - fraction;
    }
}
