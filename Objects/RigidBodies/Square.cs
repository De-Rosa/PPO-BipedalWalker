using Microsoft.Xna.Framework;
using Physics.Bodies;
using Physics.Materials;

namespace Physics.Objects.RigidBodies;

public class Square : RigidBody, IObject
{
    public float radius;

    public void Update(IObject[] objects, float deltaTime)
    {
        Step(objects, deltaTime);
    }

    private Square(IMaterial material, Skeleton skeleton, bool isStatic) : base(material, skeleton, isStatic) {}

    public static Square FromSize(IMaterial material, Vector2 centroid, float size, bool isStatic = false)
    {
        Skeleton skeleton = new Skeleton();

        float adjustment = (float) 0.5 * size;
        skeleton.AddPositions(new Position[]
        {
            new Position(centroid.X + adjustment, centroid.Y + adjustment),
            new Position(centroid.X - adjustment, centroid.Y + adjustment),
            new Position(centroid.X - adjustment, centroid.Y - adjustment),
            new Position(centroid.X + adjustment, centroid.Y - adjustment)
        });
        
        return new Square(material, skeleton, isStatic);
    }
    
    public Vector2[] GetVectors()
    {
        return _skeleton.GetVectors();
    }
    
    public Vector2 GetCentroid()
    {
        return _skeleton.GetCentroid();
    }
    
    public RigidBody GetBody()
    {
        return this;
    }
    
    public Vector2[] GetContactPoints()
    {
        return contactPoints;
    }

    public void Move(Vector2 vector)
    {
        _skeleton.Move(vector);
    }
}