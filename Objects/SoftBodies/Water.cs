using System;
using System.Collections.Generic;
using Microsoft.Xna.Framework;
using Physics.Bodies;
using Physics.Bodies.Physics;
using Physics.Materials;
using Physics.Objects.RigidBodies;

namespace Physics.Objects.SoftBodies;

public class Water
{
    private readonly float _radius;
    private Vector2 _position;
    private Vector2 _previousPosition;
    
    private Vector2 _velocity;
    private Vector2 _acceleration;

    private readonly float _inverseMass;

    public Water(Vector2 position, float radius)
    {
        _inverseMass = 0.1f;
        _radius = radius;
        _position = position;
        _previousPosition = position;
    }

    public void AddAcceleration(Vector2 acceleration)
    {
        _acceleration += acceleration;
    }

    public void AddForce(Vector2 force)
    {
        _acceleration += force * _inverseMass;
    }

    public void AddVelocity(Vector2 velocity)
    {
        _velocity += velocity;
    }

    public void Update(float deltaTime, List<IObject> rigidBodies, List<Water> water)
    {
        Step(deltaTime);
        ResolveCollisions(rigidBodies, water);
    }
    
    public void Step(float deltaTime)
    {
        StepVelocity(deltaTime);
    }

    public void ResolveCollisions(List<IObject> rigidBodies, List<Water> water)
    {
        foreach (var iObject in rigidBodies)
        {
            RigidBody iBody = (RigidBody) iObject.GetBody();
            if (!Skeleton.IsColliding(_position, iBody.GetSkeleton())) continue;
            if (!RayCasting.IsColliding(_position, iBody.GetVectors(), out var displacedPoint,
                    out var normal)) continue;
            
            _velocity -= 1.5f * Vector2.Dot(_velocity, normal) * normal;
            _previousPosition = _position;
            _position = displacedPoint;
        }
        
        foreach (var waterDrop in water)
        {
            if (waterDrop == this) continue;
            
            if (IsColliding(this, waterDrop, out Vector2 normal, out float depth))
            {
                _previousPosition = _position;
                waterDrop._previousPosition = waterDrop._position;
                
                _velocity -= 1.5f * Vector2.Dot(_velocity, normal) * normal;
                waterDrop._velocity -= 1.5f * Vector2.Dot(waterDrop._velocity, -normal) * -normal;
                
                _position -= normal * depth / 2;
                waterDrop._position += normal * depth / 2;
            }
        }
    }

    private void StepVelocity(float deltaTime)
    {
        _velocity += _acceleration * deltaTime;
        Vector2 toMove = _velocity * deltaTime;
        
        _previousPosition = _position;
        _position = _position * 2 - _previousPosition + toMove;
    }

    public Vector2 GetVector()
    {
        return _position;
    }

    public Vector2 GetVelocity()
    {
        return _velocity;
    }

    private static bool IsColliding(Water pointA, Water pointB, out Vector2 normal, out float depth)
    {
        depth = Vector2.Distance(pointA._position, pointB._position);
        normal = pointB._position - pointA._position;
        normal.Normalize();
        return depth < (pointA._radius + pointB._radius);
    }
}