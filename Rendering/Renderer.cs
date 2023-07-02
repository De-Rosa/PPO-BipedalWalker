using System.Collections.Generic;
using Physics.Objects.RigidBodies;
using System;
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using Physics.Bodies;
using Physics.Objects;
using Physics.Objects.SoftBodies;
using Point = Physics.Objects.SoftBodies.Point;

namespace Physics.Rendering;

public sealed class Renderer
{
    private readonly SpriteBatch _spriteBatch;
    private readonly Texture2D _lineTexture;

    private Vector2 _camera;
    
    public Renderer(SpriteBatch spriteBatch)
    {
        _spriteBatch = spriteBatch;
        _lineTexture = new Texture2D(spriteBatch.GraphicsDevice, 1, 1);
        _lineTexture.SetData(new[] { Color.White });
        
        _camera = Vector2.Zero;
    }

    public void MoveCamera(Vector2 vector)
    {
        _camera += vector;
    }

    public void RenderRigidObject(IObject iObject)
    {
        RigidBody iBody = (RigidBody)iObject.GetBody();
        List<Vector2> vectors = iBody.GetVectors();
        Color color = iBody.IsStatic() ? Color.White : iBody.GetColor();
        
        for (int i = 0; i < vectors.Count; i++)
        {
            DrawLine(vectors[i] + _camera, vectors[(i+1) % vectors.Count] + _camera, color);
        }
    }

    public void RenderJoint(List<Tuple<Vector2, Color>> colors)
    {
        foreach (var color in colors)
        {
            DrawSquare(color.Item1, 3, color.Item2, 2f);
        }
    }

    public void RenderSoftObject(IObject softObject)
    {
        SoftBody iBody = (SoftBody) softObject.GetBody();
        List<Point> points = iBody.GetPoints();
        foreach (var point in points)
        {
            DrawSquare(point.GetVector() + _camera, 3, Color.DarkRed, 2f);
        }

        List<Spring> springs = iBody.GetSprings();
        foreach (var spring in springs)
        {
            Vector2 pointA = spring.GetPointA().GetVector();
            Vector2 pointB = spring.GetPointB().GetVector();
            DrawLine(pointA + _camera, pointB + _camera, Color.DarkRed);
        }
    }
    
    public void RenderWater(Water water)
    {
        DrawSquare(water.GetVector() + _camera, 7, Color.Cyan, 7f);
    }

    public void DrawLine(Vector2 point1, Vector2 point2, Color color, float thickness = 1f)
    {
        var distance = Vector2.Distance(point1, point2);
        var angle = (float)Math.Atan2(point2.Y - point1.Y, point2.X - point1.X);
        DrawLine(point1 + _camera, distance, angle, color, thickness);
    }

    private void DrawLine(Vector2 point, float length, float angle, Color color, float thickness = 1f)
    {
        var origin = new Vector2(0f, 0.5f);
        var scale = new Vector2(length, thickness);
        _spriteBatch.Draw(_lineTexture, point + _camera, null, color, angle, origin, scale, SpriteEffects.None, 0);
    }

    public void DrawSquare(Vector2 origin, float length, Color color, float thickness = 1f)
    {
        var squareOrigin = new Vector2(origin.X - length / 2, origin.Y - length / 2);
        Vector2[] points = new Vector2[] { squareOrigin, new Vector2(squareOrigin.X + length, squareOrigin.Y), new Vector2(squareOrigin.X + length, squareOrigin.Y + length), new Vector2(squareOrigin.X, squareOrigin.Y + length)};
        for (int i = 0; i < points.Length; i++)
        {
            DrawLine(points[i] + _camera, points[(i + 1) % points.Length] + _camera, color, thickness);
        }
    }
}