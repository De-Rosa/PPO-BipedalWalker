using System.Collections.Generic;
using Microsoft.Xna.Framework;
using Physics.Bodies;
using Physics.Materials;

namespace Physics.Objects.RigidBodies;

// Hull class, creates an IObject from a given set of points.
public class Hull : RigidBody, IObject
{
    public void Update(List<RigidBody> rigidBodies, float deltaTime)
    {
        Step(rigidBodies, deltaTime);
    }

    private Hull(IMaterial material, Skeleton skeleton, bool isStatic, bool isFloor) : base(material, skeleton, isStatic, isFloor) {}

    public static Hull FromSkeleton(IMaterial material, Skeleton skeleton, bool isStatic = false, bool isFloor = false)
    {
        return new Hull(material, skeleton, isStatic, isFloor);
    }

    public static Hull FromPositions(IMaterial material, Vector2[] positions, bool isStatic = false, bool isFloor = false)
    {
        Skeleton skeleton = new Skeleton();
        skeleton.AddVectors(positions);

        return FromSkeleton(material, skeleton, isStatic, isFloor);
    }

    public void Rotate(float angle)
    {
        Skeleton.Rotate(angle);
    }
}