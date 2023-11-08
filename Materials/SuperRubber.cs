using Microsoft.Xna.Framework;

namespace NEA.Materials;

public class SuperRubber : IMaterial
{
    public float InverseMass { get; } = 11;
    public float Restitution { get; } = 1f;
    public float Friction { get; } = 1f;
    public Color Color { get; } = Color.Gray;

}