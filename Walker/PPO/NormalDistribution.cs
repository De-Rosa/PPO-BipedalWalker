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
}