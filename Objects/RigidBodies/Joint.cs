using System.Collections.Generic;
using Microsoft.Xna.Framework;
using NEA.Bodies;
using NEA.Bodies.Physics;

namespace NEA.Objects.RigidBodies;

// Joint class, represents a revolute joint between two rigid bodies.
public class Joint
{
    private readonly RigidBody _bodyA;
    private readonly RigidBody _bodyB;

    private readonly int _indexA;
    private readonly int _indexB;

    private float _currentTorque;

    public Joint(RigidBody bodyA, RigidBody bodyB, int pointIndexA, int pointIndexB)
    {
        _bodyA = bodyA;
        _bodyB = bodyB;
        _indexA = pointIndexA;
        _indexB = pointIndexB;

        _currentTorque = 0;
    }
    
    // Steps the joint by applying a rotation impulse.
    public void Step()
    {
        Vector2 vectorAB = GetPointB() - GetPointA();
        if (vectorAB == Vector2.Zero) return;
        float depth = vectorAB.Length();
        vectorAB.Normalize();
        _bodyA.GetSkeleton().Move(vectorAB * depth / 2);
        _bodyB.GetSkeleton().Move(-vectorAB * depth / 2);
        
        
        Impulses.ResolveJoint(_bodyB, _bodyA, new List<Vector2>(){GetPointA(), GetPointB()}, vectorAB);
    }

    // Returns the joint connection point for rigid body A.
    public Vector2 GetPointA()
    {
        return _bodyA.GetSkeleton().GetVectors()[_indexA];
    }
    
    // Returns the joint connection point for rigid body B.
    public Vector2 GetPointB()
    {
        return _bodyB.GetSkeleton().GetVectors()[_indexB];
    }

    // Sets the torque of the joint to the given value.
    public void SetTorque(float amount)
    {
        float changeInTorque = amount - _currentTorque;
        _currentTorque = amount;
        _bodyB.AddAngularVelocity(changeInTorque * 5f);
    }

    // Returns the current torque of the walker.
    public float GetTorque()
    {
        return _currentTorque;
    }
}