using Microsoft.Xna.Framework;

namespace NEA.Materials;

// Material interface, represents all materials.
public interface IMaterial
{
    public float InverseMass { get; }
    public float Friction { get; }
    public float Restitution { get; }
    public Color Color { get; }
}