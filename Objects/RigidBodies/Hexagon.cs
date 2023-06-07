using System.Collections.Generic;
using Microsoft.Xna.Framework;
using Physics.Bodies;
using Physics.Materials;

namespace Physics.Objects.RigidBodies;

public class Hexagon : RigidBody, IObject
{
    public void Update(List<IObject> rigidBodies, List<IObject> softBodies, float deltaTime)
    {
        Step(rigidBodies, deltaTime);
    }

    private Hexagon(IMaterial material, Skeleton skeleton, bool isStatic) : base(material, skeleton, isStatic) {}

    public static Hexagon FromSize(IMaterial material, Vector2 centroid, float size, bool isStatic = false)
    {
        Skeleton skeleton = new Skeleton();

        float adjustment = (float) 0.5 * size;
        skeleton.AddVectors(new Vector2[]
        {
            new Vector2(centroid.X + (adjustment * 0.5f), centroid.Y + adjustment),
            new Vector2(centroid.X - (adjustment * 0.5f), centroid.Y + adjustment),
            new Vector2(centroid.X - adjustment, centroid.Y),
            new Vector2(centroid.X - (adjustment * 0.5f), centroid.Y - adjustment),
            new Vector2(centroid.X + (adjustment * 0.5f), centroid.Y - adjustment),
            new Vector2(centroid.X + adjustment, centroid.Y)
        });
        
        return new Hexagon(material, skeleton, isStatic);
    }

    public IBody GetBody()
    {
        return this;
    }
}