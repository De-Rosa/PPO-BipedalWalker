using Microsoft.Xna.Framework;

namespace NEA.Materials;

public class Titanium : IMaterial
{
    public float InverseMass { get; set; } = 0.01f;
    public float Restitution { get; set; } = 0.1f;
    public float Friction { get; set; } = 0.2f;
    public Color Color { get; } = Color.SlateGray;

}