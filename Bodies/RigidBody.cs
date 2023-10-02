using System;
using System.Collections.Generic;
using Microsoft.Xna.Framework;
using Physics.Bodies.Physics;
using Physics.Materials;
using Physics.Objects.RigidBodies;
using Vector2 = Microsoft.Xna.Framework.Vector2;

namespace Physics.Bodies;

// Rigid body class, represents a physical object which has collision.
public class RigidBody
{
    // Rigid body variables
    protected readonly Skeleton Skeleton;
    private readonly List<RigidBody> _associatedBodies;
    private readonly bool _isStatic;
    private readonly bool _isFloor;
    private readonly Color _color;
    public bool Collided;
    
    // Physics properties
    private float _inverseMass;
    private float _inverseInertia;
    private readonly float _restitution;
    private readonly float _friction;
    
    // Physics state variables
    private Vector2 _acceleration;
    private Vector2 _linearVelocity;
    private float _angularVelocity;
    private float _torque;
    private float _angle;

    public RigidBody(IMaterial material, Skeleton skeleton, bool isStatic = false, bool isFloor = false)
    {
        Skeleton = skeleton;
        _associatedBodies = new List<RigidBody>();
        _isStatic = isStatic;
        _isFloor = isFloor;
        _color = material.Color;
        Collided = false;
        
        _restitution = material.Restitution;
        _friction = material.Friction;

        _inverseMass = isStatic ? 0f : material.InverseMass;
        _inverseInertia = isStatic ? 0f : 0.001f * material.InverseMass;
    }

    // Steps the rigid body.
    // Steps the velocities, then checks and resolves any collisions.
    protected void Step(List<RigidBody> objects, float deltaTime)
    {
        StepLinearVelocity(deltaTime);
        if (_isStatic) return;
        
        StepAngularVelocity(deltaTime);
        ResolveCollisions(objects);
    }

    // Detects and resolves collisions.
    // An AABB collision check is performed, then a SAT check.
    // If SAT returns true, we find the contact points and then resolve the collision.
    private void ResolveCollisions(List<RigidBody> objects)
    {
        foreach (var body in objects)
        {
            if (body == this) continue;
            if (_associatedBodies.Contains(body)) continue;
            
            if (!Skeleton.IsColliding(Skeleton, body.Skeleton)) continue;
            
            List<Vector2> vectorA = Skeleton.GetVectors();
            List<Vector2> vectorB = body.GetVectors();
            
            if (SATCollision.IsColliding(vectorA, vectorB, Skeleton.GetCentroid(), body.GetCentroid(), out Vector2 normal, out float depth))
            {
                if (body._isFloor) Collided = true;
                if (_isFloor) body.Collided = true;
                
                List<Vector2> contactPoints = ContactPoints.GetContactPoints(vectorA, vectorB, normal);
                MoveObjects(this, body, normal, depth);
                Impulses.ResolveCollisions(this, body, contactPoints, normal);
            }
        }
    }

    // Moves the objects so they are no longer colliding.
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
    
    // Increment the linear velocity.
    private void StepLinearVelocity(float deltaTime)
    {
        _linearVelocity += _acceleration * deltaTime;
        Skeleton.Move(_linearVelocity * deltaTime);
    }

    // Increment the angular velocity.
    private void StepAngularVelocity(float deltaTime)
    {
        _angle += _angularVelocity * deltaTime;
        _angle = WrapAngle(_angle);

        Skeleton.Rotate(_angularVelocity * deltaTime);
    }

    // Wraps an angle to [-PI, PI].
    private float WrapAngle(float angle)
    {
        return angle switch
        {
            > MathF.PI => angle - MathF.Tau,
            < -MathF.PI => angle + MathF.Tau,
            _ => angle
        };
    }
    
    // Add a rigid body to the 'no collide' list.
    public void AddAssociatedBody(RigidBody rigidBody)
    {
        _associatedBodies.Add(rigidBody);
    }
    
    // Add multiple rigid bodies ot the 'no collide' list.
    public void AddAssociatedBodies(RigidBody[] rigidBodies)
    {
        _associatedBodies.AddRange(rigidBodies);
    }

    // Add an acceleration.
    // Used for adding gravity to an object.
    public void AddAcceleration(Vector2 acceleration)
    {
        _acceleration += acceleration;
    }

    // Sets the inverse inertia for the rigid body.
    public void SetInverseInertia(float inverseInertia)
    {
        _inverseInertia = inverseInertia;
    }

    // Sets the linear velocity for the rigid body.
    public void SetLinearVelocity(Vector2 velocity)
    {
        _linearVelocity = velocity;
    }

    // Sets the angular velocity for the rigid body.
    public void SetAngularVelocity(float angularVelocity)
    {
        _angularVelocity = angularVelocity;
    }
    
    // Adds angular velocity for the rigid body.
    public void AddAngularVelocity(float angularVelocity)
    {
        _angularVelocity += angularVelocity;
    }

    // Smooths the corners of the rigid body.
    public void SmoothCorners(int count = 1)
    {
        Skeleton.SmoothCorners(count);
    }

    // Returns the linear velocity of the rigid body.
    public Vector2 GetLinearVelocity()
    {
        return _linearVelocity;
    }

    // Returns the angular velocity of the rigid body.
    public float GetAngularVelocity()
    {
        return _angularVelocity;
    }
    
    // Returns the current angle of the rigid body.
    public float GetAngle()
    {
        return _angle;
    }

    // Returns the restitution of the rigid body.
    public float GetRestitution()
    {
        return _restitution;
    }

    // Returns the friction of the rigid body.
    public float GetFriction()
    {
        return _friction;
    }
    
    // Returns the inverse mass of the rigid body.
    public float GetInverseMass()
    {
        return _inverseMass;
    }
    
    // Returns the inverse inertia of the rigid body.
    public float GetInverseInertia()
    {
        return _inverseInertia;
    }

    // Returns the centroid of the rigid body.
    public Vector2 GetCentroid()
    {
        return Skeleton.GetCentroid();
    }

    // Returns the skeleton of the rigid body.
    public Skeleton GetSkeleton()
    {
        return Skeleton;
    }

    // Returns the color of the rigid body.
    public Color GetColor()
    {
        return _color;
    }

    // Returns the vectors of the rigid body.
    public List<Vector2> GetVectors()
    {
        return Skeleton.GetVectors();
    }

    // Returns if the rigid body is static.
    public bool IsStatic()
    {
        return _isStatic;
    }
}