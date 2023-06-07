using System;
using System.Collections.Generic;
using Microsoft.Xna.Framework;
using Physics.Bodies;
using Physics.Bodies.Physics;
using Physics.Materials;
using Physics.Objects.RigidBodies;

namespace Physics.Objects.SoftBodies;

public class Point
{
    private readonly float _radius;
    private Vector2 _position;
    private Vector2 _previousPosition;
    
    private Vector2 _velocity;
    private Vector2 _acceleration;

    private readonly float _inverseMass;

    public Point(IMaterial material, Vector2 position, float radius)
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
    
    public void Step(float deltaTime)
    {
        StepVelocity(deltaTime);
    }

    public void ResolveCollisions(List<IObject> rigidBodies, List<IObject> softBodies, List<Point> points, SoftBody parent)
    {
        foreach (var iObject in rigidBodies)
        {
            RigidBody iBody = (RigidBody) iObject.GetBody();
            if (!Skeleton.IsColliding(_position, iBody.GetSkeleton())) continue;
            if (!RayCasting.IsColliding(_position, iBody.GetVectors(), out var displacedPoint,
                    out var normal)) continue;
            
            _velocity -= 2 * Vector2.Dot(_velocity, normal) * normal;
            _previousPosition = _position;
            _position = displacedPoint;
        }

        foreach (var point in points)
        {
            if (point == this) continue;
            if (IsColliding(this, point, out Vector2 normal, out float depth))
            {
                _previousPosition = _position;
                point._previousPosition = point._position;
                
                _velocity -= 2 * Vector2.Dot(_velocity, normal) * normal;
                point._velocity -= 2 * Vector2.Dot(point._velocity, -normal) * -normal;
                
                _position -= normal * depth / 2;
                point._position += normal * depth / 2;
            }
        }
        
        foreach (var softBody in softBodies)
        {
            SoftBody iBody = (SoftBody)softBody;
            if (iBody == parent) continue;
            
            List<Point> bodyPoints = iBody.GetPoints();
            
            foreach (var point in bodyPoints)
            {
                if (IsColliding(this, point, out Vector2 normal, out float depth))
                {
                    _previousPosition = _position;
                    point._previousPosition = point._position;
                
                    _velocity -= 2 * Vector2.Dot(_velocity, normal) * normal;
                    point._velocity -= 2 * Vector2.Dot(point._velocity, -normal) * -normal;
                
                    _position -= normal * depth / 2;
                    point._position += normal * depth / 2;
                }
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

    private static bool IsColliding(Point pointA, Point pointB, out Vector2 normal, out float depth)
    {
        depth = Vector2.Distance(pointA._position, pointB._position);
        normal = pointB._position - pointA._position;
        normal.Normalize();
        return depth < (pointA._radius + pointB._radius);
    }
}