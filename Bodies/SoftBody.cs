using System.Collections.Generic;
using Microsoft.Xna.Framework;
using Physics.Objects;
using Physics.Objects.SoftBodies;
using Point = Physics.Objects.SoftBodies.Point;

namespace Physics.Bodies;

// use bunch of points for water?
public class SoftBody : IBody
{
    private List<Point> _points;
    private List<Spring> _springs;

    public SoftBody(List<Point> points, List<Spring> springs)
    {
        _points = points;
        _springs = springs;
    }

    protected void Step(List<IObject> rigidBodies, List<IObject> softBodies, float deltaTime)
    {
        StepPoints(deltaTime);
        ResolveCollisions(rigidBodies, softBodies);
        StepSprings();
    }

    public void AddAcceleration(Vector2 acceleration)
    {
        foreach (var point in _points)
        {
            point.AddAcceleration(acceleration);
        }
    }
    
    private void StepPoints(float deltaTime)
    {
        foreach (var point in _points)
        {
            point.Step(deltaTime);
        }
    }
    
    private void StepSprings()
    {
        foreach (var spring in _springs)
        { 
            spring.Step();
        }
    }

    private void ResolveCollisions(List<IObject> rigidBodies, List<IObject> softBodies)
    {
        foreach (var point in _points)
        {
            point.ResolveCollisions(rigidBodies, softBodies, _points, this);
        }
    }

    public List<Point> GetPoints()
    {
        return _points;
    }
    
    public List<Spring> GetSprings()
    {
        return _springs;
    }
}