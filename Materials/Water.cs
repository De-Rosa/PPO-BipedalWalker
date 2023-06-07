using Microsoft.Xna.Framework;

namespace Physics.Materials;

public class Water : IMaterial
{
    public float InverseMass { get; set; } = 10;
    public float Restitution { get; set; } = 0.5f;
    public float Friction { get; set; } = 0.5f;
    public Color Color { get; } = Color.LightBlue;
}