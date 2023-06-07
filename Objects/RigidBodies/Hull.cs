using System.Collections.Generic;
using Microsoft.Xna.Framework;
using Physics.Bodies;
using Physics.Materials;

namespace Physics.Objects.RigidBodies;

public class Hull : RigidBody, IObject
{
    public void Update(List<IObject> rigidBodies, List<IObject> softBodies, float deltaTime)
    {
        Step(rigidBodies, deltaTime);
    }

    private Hull(IMaterial material, Skeleton skeleton, bool isStatic,  bool isFragile, bool isFloor) : base(material, skeleton, isStatic, isFragile, isFloor) {}

    public static Hull FromSkeleton(IMaterial material, Skeleton skeleton, bool isStatic = false, bool isFragile = false, bool isFloor = false)
    {
        return new Hull(material, skeleton, isStatic, isFragile, isFloor);
    }

    public static Hull FromPositions(IMaterial material, Vector2[] positions, bool isStatic = false, bool isFragile = false, bool isFloor = false)
    {
        Skeleton skeleton = new Skeleton();
        skeleton.AddVectors(positions);

        return FromSkeleton(material, skeleton, isStatic, isFragile, isFloor);
    }
    
    public IBody GetBody()
    {
        return this;
    }

    public void Rotate(float angle)
    {
        Skeleton.Rotate(angle);
    }
}