using Microsoft.Xna.Framework;

namespace Physics.Materials;

public class Wood : IMaterial
{
    public float InverseMass { get; set; } = 20;
    public float Restitution { get; set; } = 0.3f;
    public float Friction { get; set; } = 0.01f;
    public Color Color { get; } = Color.SandyBrown;

}