using System.Collections.Generic;
using Microsoft.Xna.Framework;
using Physics.Bodies;
using Physics.Materials;

namespace Physics.Objects.RigidBodies;

public class Triangle : RigidBody, IObject
{
    public void Update(List<IObject> rigidBodies, List<IObject> softBodies, float deltaTime)
    {
        Step(rigidBodies, deltaTime);
    }

    private Triangle(IMaterial material, Skeleton skeleton, bool isStatic) : base(material, skeleton, isStatic) {}

    public static Triangle FromSize(IMaterial material, Vector2 centroid, float size, bool isStatic = false)
    {
        Skeleton skeleton = new Skeleton();

        float adjustment = (float) 0.5 * size;
        skeleton.AddVectors(new Vector2[]
        {
            new Vector2(centroid.X, centroid.Y + adjustment),
            new Vector2(centroid.X - adjustment, centroid.Y - adjustment),
            new Vector2(centroid.X + adjustment, centroid.Y - adjustment),
        });
        
        return new Triangle(material, skeleton, isStatic);
    }
    
    public List<Vector2> GetVectors()
    {
        return Skeleton.GetVectors();
    }

    public IBody GetBody()
    {
        return this;
    }
}