using System.Collections.Generic;
using Physics.Objects.RigidBodies;
using System;
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using Physics.Bodies;
using Physics.Objects;
using Matrix = Physics.Walker.PPO.Matrix;

namespace Physics.Rendering;

public sealed class Renderer
{
    private readonly SpriteBatch _spriteBatch;
    private readonly Texture2D _lineTexture;
    private readonly ConsoleRenderer _consoleRenderer;

    private Vector2 _camera;
    
    public Renderer(SpriteBatch spriteBatch)
    {
        _spriteBatch = spriteBatch;
        _lineTexture = new Texture2D(spriteBatch.GraphicsDevice, 1, 1);
        _lineTexture.SetData(new[] { Color.White });

        _consoleRenderer = new ConsoleRenderer();
        
        _camera = Vector2.Zero;
    }

    public void UpdateConsole()
    {
        _consoleRenderer.Update();
    }

    public void UpdateConsole(int episode, int timeStep, float distance, float averageReward, float bestDistance, float pastAverageReward, Matrix state)
    {
        _consoleRenderer.Update(episode, timeStep, distance, averageReward, bestDistance, pastAverageReward, state);
    }

    public void UpdateConsole(int epoch, int batch, int batchSize, float criticLoss)
    {
        _consoleRenderer.Update(epoch, batch, batchSize, criticLoss);
    }

    public void ExitTraining()
    {
        _consoleRenderer.ExitTraining();
    }
    
    public void AddAverageEpisodeReward(float reward)
    {
        _consoleRenderer.AddAverageEpisodeReward(reward);
    }

    public void MoveCamera(Vector2 vector)
    {
        _camera += vector;
    }

    public void RenderRigidObject(RigidBody body)
    {
        List<Vector2> vectors = body.GetVectors();
        Color color = body.IsStatic() ? Color.White : body.GetColor();
        
        for (int i = 0; i < vectors.Count; i++)
        {
            DrawLine(vectors[i] + _camera, vectors[(i+1) % vectors.Count] + _camera, color);
        }
    }

    public void RenderJoints(List<Tuple<Vector2, Color>> colors)
    {
        foreach (var color in colors)
        {
            DrawSquare(color.Item1, 3, color.Item2, 2f);
        }
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