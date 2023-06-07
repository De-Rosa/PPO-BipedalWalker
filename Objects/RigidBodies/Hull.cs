using Microsoft.Xna.Framework;
using Physics.Bodies;
using Physics.Materials;

namespace Physics.Objects.RigidBodies;

public class Hull : RigidBody, IObject
{
    public void Update(IObject[] objects, float deltaTime)
    {
        Step(objects, deltaTime);
    }

    private Hull(IMaterial material, Skeleton skeleton, bool isStatic) : base(material, skeleton, isStatic) {}

    public static Hull FromSkeleton(IMaterial material, Skeleton skeleton, bool isStatic = false)
    {
        return new Hull(material, skeleton, isStatic);
    }

    public static Hull FromPositions(IMaterial material, Position[] positions, bool isStatic = false)
    {
        Skeleton skeleton = new Skeleton();
        skeleton.AddPositions(positions);

        return FromSkeleton(material, skeleton, isStatic);
    }

    public Vector2[] GetVectors()
    {
        return _skeleton.GetVectors();
    }

    public Vector2[] GetContactPoints()
    {
        return contactPoints;
    }
    
    public Vector2 GetCentroid()
    {
        return _skeleton.GetCentroid();
    }

    public RigidBody GetBody()
    {
        return this;
    }
    
    public void Move(Vector2 vector)
    {
        _skeleton.Move(vector);
    }
}