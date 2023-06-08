using Microsoft.Xna.Framework;

namespace Physics.Materials;

public class Carpet : IMaterial
{
    public float InverseMass { get; set; } = 10;
    public float Restitution { get; set; } = 0.3f;
    public float Friction { get; set; } = 0.8f;
    public Color Color { get; } = Color.DarkCyan;

}