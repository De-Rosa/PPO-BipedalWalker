using Microsoft.Xna.Framework;

namespace Physics.Objects.Maths;

public class VectorMath
{
    // finds the mean of group of points
    public static Vector2 FindCentroid(Vector2[] vectors)
    {
        int count = vectors.Length;
        Vector2 sum = Vector2.Zero;

        foreach (var vector in vectors)
        {
            sum += vector;
        }

        sum /= count;

        return sum;
    }
}