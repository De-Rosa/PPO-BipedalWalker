using Microsoft.Xna.Framework;

namespace NEA.Materials;

public class Paper : IMaterial
{
    public float InverseMass { get; set; } = 1;
    public float Restitution { get; set; } = 0.3f;
    public float Friction { get; set; } = 0.1f;
    public Color Color { get; } = Color.Gray;

}