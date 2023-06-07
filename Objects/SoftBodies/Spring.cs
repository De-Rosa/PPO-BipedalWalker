using System;
using Microsoft.Xna.Framework;

namespace Physics.Objects.SoftBodies;

public class Spring
{
    private readonly Point _pointA;
    private readonly Point _pointB;
    
    private const float Stiffness = 1f;
    private const float Damping = 0.2f;

    private readonly float _restLength;

    public Spring(Point pointA, Point pointB)
    {
        _pointA = pointA;
        _pointB = pointB;

        _restLength = Vector2.Distance(_pointA.GetVector(), _pointB.GetVector());
    }

    public void Step()
    {
        Vector2 vectorAB = _pointB.GetVector() - _pointA.GetVector();
        Vector2 vectorBA = -vectorAB;
        
        // Hooke's law
        float force = Stiffness * (vectorAB.Length() - _restLength);
        force += GetDampForce();
        
        _pointA.AddVelocity(force * Vector2.Normalize(vectorAB));
        _pointB.AddVelocity(force * Vector2.Normalize(vectorBA));
    }

    private float GetDampForce()
    {
        Vector2 vectorAB = _pointB.GetVector() - _pointA.GetVector();
        vectorAB.Normalize();
        
        Vector2 velocity = _pointB.GetVelocity() - _pointA.GetVelocity();
        
        float velocityDotAB = Vector2.Dot(velocity, vectorAB);

        return velocityDotAB * Damping;
    }

    public Point GetPointA()
    {
        return _pointA;
    }

    public Point GetPointB()
    {
        return _pointB;
    }
}