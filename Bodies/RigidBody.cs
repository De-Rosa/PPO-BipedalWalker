using System;
using System.Collections.Generic;
using System.Diagnostics.Metrics;
using System.Numerics;
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Audio;
using Physics.Bodies.Physics;
using Physics.Materials;
using Physics.Objects;
using Physics.Objects.RigidBodies;
using Vector2 = Microsoft.Xna.Framework.Vector2;

namespace Physics.Bodies;

public class RigidBody : IBody
{
    protected readonly Skeleton Skeleton;
    private readonly List<RigidBody> _associatedBodies;
    private readonly bool _isStatic;
    private readonly Color _color;
    
    private readonly float _inverseMass;
    private readonly float _inverseInertia;
    
    private readonly float _restitution;
    private readonly float _friction;
    
    private Vector2 _acceleration;
    private Vector2 _linearVelocity;
    
    private float _angularVelocity;
    private float _angle;

    private readonly bool _fragile;
    private readonly bool _isFloor;
    public bool Broken;

    public RigidBody(IMaterial material, Skeleton skeleton, bool isStatic = false, bool isFragile = false, bool isFloor = false)
    {
        Skeleton = skeleton;

        _restitution = material.Restitution;
        _friction = material.Friction;
        _associatedBodies = new List<RigidBody>();
        _color = material.Color;
        
        _fragile = isFragile;
        _isFloor = isFloor;
        Broken = false;
        
        _isStatic = isStatic;
        
        if (isStatic)
        {
            _inverseMass = 0f;
            _inverseInertia = 0f;
        }
        else
        {
            _inverseMass = material.InverseMass;
            _inverseInertia = 0.001f * material.InverseMass;
        }
    }

    protected void Step(List<IObject> objects, float deltaTime)
    {
        StepLinearVelocity(deltaTime);

        if (_isStatic) return;
        
        StepAngularVelocity(deltaTime);
        ResolveCollisions(objects, deltaTime);
    }

    private void ResolveCollisions(List<IObject> objects, float deltaTime)
    {
        foreach (var iObject in objects)
        {
            if (iObject == this) continue;
            RigidBody iBody = (RigidBody) iObject.GetBody();
            if (_associatedBodies.Contains(iBody)) continue;
            
            if (!Skeleton.IsColliding(Skeleton, iBody.Skeleton)) continue;
            
            List<Vector2> vectorA = Skeleton.GetVectors();
            List<Vector2> vectorB = iBody.GetVectors();
            
            if (SATCollision.IsColliding(vectorA, vectorB, Skeleton.GetCentroid(), iBody.GetCentroid(), out Vector2 normal, out float depth))
            {
                if (iBody._isFloor && _fragile) Broken = true;
                List<Vector2> contactPoints = ContactPoints.GetContactPoints(vectorA, vectorB, normal);
                MoveObjects(this, iBody, normal, depth);
                Impulses.ResolveCollisions(this, iBody, contactPoints, normal);
            }
        }
    }

    private static void MoveObjects(RigidBody objectA, RigidBody objectB, Vector2 normal, float depth)
    {
        if (objectA._isStatic)
        {
            objectB.Skeleton.Move(-normal * depth);
        } else if (objectB._isStatic)
        {
            objectA.Skeleton.Move(normal * depth);
        }
        else
        {
            objectA.Skeleton.Move(normal * depth / 2);
            objectB.Skeleton.Move(-normal * depth / 2);
        }
    }

    private void StepLinearVelocity(float deltaTime)
    {
        _linearVelocity += _acceleration * deltaTime;
        Skeleton.Move(_linearVelocity * deltaTime);
    }

    private void StepAngularVelocity(float deltaTime)
    {
        _angle = _angularVelocity * deltaTime;
        Skeleton.Rotate(_angle);
    }
    
    public void AddAcceleration(Vector2 acceleration)
    {
        _acceleration += acceleration;
    }

    public void SetLinearVelocity(Vector2 velocity)
    {
        _linearVelocity = velocity;
    }
    
    public void AddLinearVelocity(Vector2 velocity)
    {
        _linearVelocity += velocity;
    }

    public void SetAngularVelocity(float angularVelocity)
    {
        _angularVelocity = angularVelocity;
    }
    
    public void AddAngularVelocity(float angularVelocity)
    {
        _angularVelocity += angularVelocity;
    }

    public void SmoothCorners(int count = 1)
    {
        Skeleton.SmoothCorners(count);
    }

    public Vector2 GetLinearVelocity()
    {
        return _linearVelocity;
    }

    public float GetAngularVelocity()
    {
        return _angularVelocity;
    }
    
    public float GetAngle()
    {
        return _angle;
    }

    public float GetRestitution()
    {
        return _restitution;
    }

    public float GetFriction()
    {
        return _friction;
    }
    
    public float GetInverseMass()
    {
        return _inverseMass;
    }
    
    public float GetInverseInertia()
    {
        return _inverseInertia;
    }

    public Vector2 GetCentroid()
    {
        return Skeleton.GetCentroid();
    }

    public Skeleton GetSkeleton()
    {
        return Skeleton;
    }

    public Color GetColor()
    {
        return _color;
    }

    public List<Vector2> GetVectors()
    {
        return Skeleton.GetVectors();
    }

    public bool IsStatic()
    {
        return _isStatic;
    }

    public void AddAssociatedBody(RigidBody rigidBody)
    {
        _associatedBodies.Add(rigidBody);
    }
    
    public void AddAssociatedBodies(RigidBody[] rigidBodies)
    {
        _associatedBodies.AddRange(rigidBodies);
    }
}