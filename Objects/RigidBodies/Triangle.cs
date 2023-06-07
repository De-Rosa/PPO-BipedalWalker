using System.Collections.Generic;
using Microsoft.Xna.Framework;
using Physics.Bodies;
using Physics.Materials;

namespace Physics.Objects.RigidBodies;

public class Square : RigidBody, IObject
{
    public void Update(List<IObject> rigidBodies, List<IObject> softBodies, float deltaTime)
    {
        Step(rigidBodies, deltaTime);
    }

    private Square(IMaterial material, Skeleton skeleton, bool isStatic) : base(material, skeleton, isStatic) {}

    public static Square FromSize(IMaterial material, Vector2 centroid, float size, bool isStatic = false)
    {
        Skeleton skeleton = new Skeleton();

        float adjustment = (float) 0.5 * size;
        skeleton.AddVectors(new Vector2[]
        {
            new Vector2(centroid.X + adjustment, centroid.Y + adjustment),
            new Vector2(centroid.X - adjustment, centroid.Y + adjustment),
            new Vector2(centroid.X - adjustment, centroid.Y - adjustment),
            new Vector2(centroid.X + adjustment, centroid.Y - adjustment)
        });
        
        return new Square(material, skeleton, isStatic);
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