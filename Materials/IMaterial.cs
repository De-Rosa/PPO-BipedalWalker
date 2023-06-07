using Microsoft.Xna.Framework;

namespace Physics.Materials;

public interface IMaterial
{
    public float InverseMass { get; }
    public float Friction { get; }
    public float Restitution { get; }
    public Color Color { get; }
}