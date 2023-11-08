using Microsoft.Xna.Framework;

namespace NEA.Materials;

public class Metal : IMaterial
{
    public float InverseMass { get; } = 15;
    public float Restitution { get; } = 0.3f;
    public float Friction { get;  } = 1f;
    public Color Color { get; } = Color.White;
}