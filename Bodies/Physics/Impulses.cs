using System;
using System.Collections.Generic;
using Vector2 = Microsoft.Xna.Framework.Vector2;

namespace Physics.Bodies.Physics;

public class Impulses
{
    // http://www.chrishecker.com/images/e/e7/Gdmphys3.pdf
    // https://research.ncl.ac.uk/game/mastersdegree/gametechnologies/previousinformation/physics6collisionresponse/2017%20Tutorial%206%20-%20Collision%20Response.pdf
    public static void ResolveCollisions(RigidBody bodyA, RigidBody bodyB, List<Vector2> contactPoints, Vector2 normal)
    {
        if (contactPoints.Count == 0) return;
        
        float restitution = Math.Max(bodyA.GetRestitution(), bodyB.GetRestitution());
        float friction = Math.Min(bodyA.GetFriction(), bodyB.GetFriction()) * 0.5f;
        
        // midpoint between both contact points
        Vector2 contact = contactPoints.Count == 2 ? Vector2.Divide(contactPoints[0] + contactPoints[1], 2) : contactPoints[0];
        Manifold manifold = new Manifold(bodyA, bodyB, normal, restitution, friction);
        
        ResolveCollisionAtPoint(manifold, contact, out Vector2 centreContactA, out Vector2 centreContactB, out float impulse);
        ResolveFrictionAtPoint(manifold, contact, out Vector2 centreContactAF, out Vector2 centreContactBF, out float impulseF, out Vector2 tangent);
        
        ApplyImpulses(manifold, manifold.Normal, impulse, centreContactA, centreContactB);
        ApplyImpulses(manifold, tangent, impulseF, centreContactAF, centreContactBF);
    }
    
    public static void ResolveJoint(RigidBody bodyA, RigidBody bodyB, List<Vector2> contactPoints, Vector2 normal)
    {
        if (contactPoints.Count == 0) return;

        Vector2 contact = contactPoints.Count == 2 ? Vector2.Divide(contactPoints[0] + contactPoints[1], 2) : contactPoints[0];
        Manifold manifold = new Manifold(bodyA, bodyB, normal, 1f, 0f);
        
        ResolveCollisionAtPoint(manifold, contact, out Vector2 centreContactA, out Vector2 centreContactB, out float impulse);
        ApplyImpulses(manifold, manifold.Normal, impulse, centreContactA, centreContactB);
    }

    private static void ResolveCollisionAtPoint(Manifold manifold, Vector2 contactPoint, out Vector2 centreContactA, out Vector2 centreContactB, out float impulse)
    {
        CalculateImpulse(manifold, contactPoint, (1 + manifold.Restitution), manifold.Normal, out centreContactA, out centreContactB, out impulse);

    }
    
    private static void ResolveFrictionAtPoint(Manifold manifold, Vector2 contactPoint, out Vector2 centreContactA, out Vector2 centreContactB, out float impulse, out Vector2 tangent)
    {
        tangent = new Vector2(-manifold.Normal.Y, manifold.Normal.X);
        CalculateImpulse(manifold, contactPoint, manifold.Friction, tangent, out centreContactA, out centreContactB, out impulse);
    }
    
    public static void ApplyImpulses(Manifold manifold, Vector2 normal, float impulse, Vector2 centreContactA,
        Vector2 centreContactB)
    {
        Vector2 impulseNormal = impulse * normal;

        // Equation 8a
        Vector2 velocityA = manifold.BodyA.GetLinearVelocity() -
                            impulseNormal * manifold.BodyA.GetInverseMass();
        Vector2 velocityB = manifold.BodyB.GetLinearVelocity() +
                            impulseNormal * manifold.BodyB.GetInverseMass();

        manifold.BodyA.SetLinearVelocity(velocityA);
        manifold.BodyB.SetLinearVelocity(velocityB);

        // Equation 8b
        Vector2 perpCentreContactA = new Vector2(-centreContactA.Y, centreContactA.X);
        float angularVelocityA = manifold.BodyA.GetAngularVelocity() -
                                 (Vector2.Dot(perpCentreContactA, impulseNormal) * manifold.BodyA.GetInverseInertia());

        Vector2 perpCentreContactB = new Vector2(-centreContactB.Y, centreContactB.X);
        float angularVelocityB = manifold.BodyB.GetAngularVelocity() +
                                 (Vector2.Dot(perpCentreContactB, impulseNormal) * manifold.BodyB.GetInverseInertia());
        
        manifold.BodyA.SetAngularVelocity(angularVelocityA);
        manifold.BodyB.SetAngularVelocity(angularVelocityB);
    }
    
    private static void CalculateImpulse(Manifold manifold, Vector2 contactPoint, float force, Vector2 normal,
        out Vector2 centreContactA, out Vector2 centreContactB, out float impulse)
    {
        centreContactA = contactPoint - manifold.BodyA.GetCentroid();
        Vector2 perpCentreContactA = new Vector2(-centreContactA.Y, centreContactA.X);
        float ctcDotNormalA = Vector2.Dot(normal, perpCentreContactA);
            
        centreContactB = contactPoint - manifold.BodyB.GetCentroid();
        Vector2 perpCentreContactB = new Vector2(-centreContactB.Y, centreContactB.X);
        float ctcDotNormalB = Vector2.Dot(normal, perpCentreContactB);

        // Equation 7
        Vector2 aVelocity = manifold.BodyA.GetLinearVelocity() +
                            (perpCentreContactA * manifold.BodyA.GetAngularVelocity());
        Vector2 bVelocity = manifold.BodyB.GetLinearVelocity() +
                            (perpCentreContactB * manifold.BodyB.GetAngularVelocity());

        // Equation 1
        Vector2 velocity =  bVelocity - aVelocity;
        
        float velocityDotNormal = Vector2.Dot(velocity, normal);
        
        // Equation 9
        impulse = -force * (velocityDotNormal);
        float denominator = (manifold.BodyA.GetInverseMass() + manifold.BodyB.GetInverseMass()) +
                            ((ctcDotNormalA * ctcDotNormalA) * manifold.BodyA.GetInverseInertia()) +
                            ((ctcDotNormalB * ctcDotNormalB) * manifold.BodyB.GetInverseInertia());
        
        impulse /= denominator;
    }

    public struct Manifold
    {
        public readonly RigidBody BodyA;
        public readonly RigidBody BodyB;
        public readonly Vector2 Normal;
        public readonly float Restitution;
        public readonly float Friction;

        public Manifold(RigidBody bodyA, RigidBody bodyB, Vector2 normal, float restitution, float friction)
        {
            BodyA = bodyA;
            BodyB = bodyB;
            Normal = normal;
            Restitution = restitution;
            Friction = friction;
        }
    }
}