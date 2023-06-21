using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.Xna.Framework;
using Physics.Bodies;
using Physics.Materials;
using Physics.Objects;
using Physics.Objects.RigidBodies;
using Physics.Rendering;
using Physics.Walker.PPO;
using Matrix = Physics.Walker.PPO.Matrix;

namespace Physics.Walker;

public class Walker
{
    private readonly IMaterial _material;
    private readonly List<Joint> _joints;
    private Vector2 _position;
    private Vector2 _previousPosition;
    private BodyParts _bodyParts;
    private readonly PPOAgent _brain;

    public bool Terminal;
    private int _tag;

    public Walker()
    {
        _joints = new List<Joint>();
        _brain = new PPO.PPOAgent(12, 4);
        _bodyParts = new BodyParts();
        _material = new Carpet();
        _position = new Vector2(125, 800);
        _previousPosition = _position;
        Terminal = false;
        _tag = 0;
    }

    public Walker(PPOAgent brain, int tag)
    {
        _joints = new List<Joint>();
        _brain = brain;
        _bodyParts = new BodyParts();
        _material = new Carpet();
        _position = new Vector2(125, 800);
        _previousPosition = _position;
        Terminal = false;
        _tag = tag;
    } 
    
    public void CreateCreature(List<IObject> rigidBodies)
    {
        CreateBodies(rigidBodies);
        CreateJoints();
        AddAssociatedBodies();
        AddAcceleration(new Vector2(0, 980));
    }

    public void Update()
    {
        _previousPosition = _position;
        _position = _joints[0].GetPointA();
        if (_bodyParts.Body.Collided) Terminal = true;
    }

    public Matrix GetActions(Matrix state, out Matrix probabilities, out Matrix mean, out Matrix std)
    {
        Matrix actions = _brain.SampleActions(state, out probabilities, out mean, out std);
        actions = Matrix.Clip(actions, 1, -1);
        return actions;
    }

    public void TakeActions(Matrix actions)
    {
        for (int i = 0; i < _joints.Count; i++)
        {
            float torque = actions.GetValue(i, 0);
            _joints[i].SetTorque(torque);
        }
    }

    public void Train(Trajectory trajectory)
    {
        _brain.Train(trajectory);
    }

    public void Save(string criticFileLocation, string muFileLocation)
    {
        _brain.Save(criticFileLocation, muFileLocation);
    }

    public void Load(string criticFileLocation, string muFileLocation)
    {
        _brain.Load(criticFileLocation, muFileLocation);
    }

    public void Render(Renderer renderer)
    {
        _brain.Render(renderer);
    }

    public List<Tuple<Vector2, Color>> GetJointColors()
    {
        List<Tuple<Vector2, Color>> colors = new List<Tuple<Vector2, Color>>();
        
        foreach (var joint in _joints)
        {
            Vector2 position = (joint.GetPointA() + joint.GetPointB()) / 2f;
            float value = (joint.GetTorque() / 15f);
            Color color = new Color(value * 255, 0, (1 - value) * 255, 100);
            colors.Add(new Tuple<Vector2, Color>(position, color));
        }

        return colors;
    }

    public List<Joint> GetJoints()
    {
        return _joints;
    }

    public Vector2 GetPosition()
    {
        return _position;
    }

    public PPOAgent GetBrain()
    {
        return _brain;
    }

    public float GetBodyVelocityMagnitude()
    {
        return _bodyParts.Body.GetLinearVelocity().Length();
    }
    public Vector2 GetChangeInPosition()
    {
        return _position - _previousPosition;
    }

    public Matrix GetState()
    {
        float[] values = new[]
        {
            _bodyParts.LeftLegLowerSegment.GetAngularVelocity() * 10,
            _bodyParts.LeftLegUpperSegment.GetAngularVelocity() * 10,
            _bodyParts.RightLegLowerSegment.GetAngularVelocity() * 10,
            _bodyParts.RightLegUpperSegment.GetAngularVelocity() * 10, 

            -_bodyParts.LeftLegLowerSegment.GetLinearVelocity().Y,
            -_bodyParts.LeftLegUpperSegment.GetLinearVelocity().Y,
            -_bodyParts.RightLegLowerSegment.GetLinearVelocity().Y,
            -_bodyParts.RightLegUpperSegment.GetLinearVelocity().Y, 
                
            _bodyParts.LeftLegLowerSegment.GetLinearVelocity().X,
            _bodyParts.LeftLegUpperSegment.GetLinearVelocity().X,
            _bodyParts.RightLegLowerSegment.GetLinearVelocity().X,
            _bodyParts.RightLegUpperSegment.GetLinearVelocity().X
        };

        return Matrix.FromValues(values);
    }

    private void NormalizeState(float[] state)
    {
        float upperBound = 1;
        float lowerBound = -1;

        for (int i = 0; i < state.Length; i++)
        {
            state[i] -= state.Min();
            state[i] /= state.Max() - state.Min() + 1e-10f;
            state[i] *= (upperBound - lowerBound);
            state[i] += lowerBound;
        }
    }

    private void CreateBodies(List<IObject> rigidBodies)
    {
        Skeleton bodySkeleton = new Skeleton();
        bodySkeleton.AddVectors(new Vector2[]
        {
            new Vector2(_position.X + 20, _position.Y + 20), // bottom right
            new Vector2(_position.X, _position.Y + 20), // bottom middle
            new Vector2(_position.X - 20, _position.Y + 20), // bottom left
            new Vector2(_position.X - 20, _position.Y - 20), // top left
            new Vector2(_position.X + 20, _position.Y - 20) // top right
        });

        _bodyParts.Body = Hull.FromSkeleton(_material, bodySkeleton);
        _bodyParts.Body.SetInverseInertia(0.0005f);
        _bodyParts.Body.SetTag(_tag);

        _bodyParts.LeftLegUpperSegment = Pole.FromSize(_material, _position + new Vector2(0, 20), 75);
        _bodyParts.LeftLegLowerSegment = Pole.FromSize(_material, _position + new Vector2(0, 50f), 75);
        _bodyParts.LeftLegUpperSegment.SetTag(_tag);
        _bodyParts.LeftLegLowerSegment.SetTag(_tag);

        _bodyParts.RightLegUpperSegment = Pole.FromSize(_material, _position + new Vector2(0, 20), 75);
        _bodyParts.RightLegLowerSegment = Pole.FromSize(_material, _position + new Vector2(0, 50f), 75);
        _bodyParts.RightLegUpperSegment.SetTag(_tag);
        _bodyParts.RightLegLowerSegment.SetTag(_tag);


        rigidBodies.AddRange(new IObject[] {_bodyParts.LeftLegLowerSegment, _bodyParts.LeftLegUpperSegment, _bodyParts.Body, _bodyParts.RightLegLowerSegment, _bodyParts.RightLegUpperSegment});
    }

    private void CreateJoints()
    {
        Joint bodyJointLeft = new Joint(_bodyParts.Body, _bodyParts.LeftLegUpperSegment, 1, 4);
        Joint bodyJointRight = new Joint(_bodyParts.Body, _bodyParts.RightLegUpperSegment, 1, 4);
        Joint leftJoint = new Joint(_bodyParts.LeftLegUpperSegment, _bodyParts.LeftLegLowerSegment, 2, 3);
        Joint rightJoint = new Joint(_bodyParts.RightLegUpperSegment, _bodyParts.RightLegLowerSegment, 2, 3);

        _joints.AddRange(new []{bodyJointLeft, bodyJointRight,  leftJoint, rightJoint});
    }

    private void AddAcceleration(Vector2 acceleration)
    {
        _bodyParts.LeftLegUpperSegment.AddAcceleration(acceleration);
        _bodyParts.LeftLegLowerSegment.AddAcceleration(acceleration);
        _bodyParts.RightLegUpperSegment.AddAcceleration(acceleration);
        _bodyParts.RightLegLowerSegment.AddAcceleration(acceleration);
        _bodyParts.Body.AddAcceleration(acceleration);
    }

    private void AddAssociatedBodies()
    {
        _bodyParts.LeftLegUpperSegment.AddAssociatedBodies(new RigidBody[] {_bodyParts.RightLegUpperSegment, _bodyParts.RightLegLowerSegment, _bodyParts.Body});
        _bodyParts.LeftLegLowerSegment.AddAssociatedBodies(new RigidBody[] {_bodyParts.RightLegUpperSegment, _bodyParts.RightLegLowerSegment, _bodyParts.Body});
        _bodyParts.RightLegUpperSegment.AddAssociatedBodies(new RigidBody[] {_bodyParts.LeftLegUpperSegment, _bodyParts.LeftLegLowerSegment, _bodyParts.Body});
        _bodyParts.RightLegLowerSegment.AddAssociatedBodies(new RigidBody[] {_bodyParts.LeftLegUpperSegment, _bodyParts.LeftLegLowerSegment, _bodyParts.Body});
        _bodyParts.Body.AddAssociatedBodies(new RigidBody[] {_bodyParts.LeftLegUpperSegment, _bodyParts.RightLegUpperSegment});
    }

    public void Reset(List<IObject> rigidBodies)
    {
        RemoveRigidObjects(rigidBodies);        
        _joints.Clear();
        _bodyParts = new BodyParts();
        Terminal = false;
        
        _position = new Vector2(125, 800);
        _previousPosition = _position;
    }

    private void RemoveRigidObjects(List<IObject> rigidBodies)
    {
        rigidBodies.Remove(_bodyParts.Body);
        rigidBodies.Remove(_bodyParts.LeftLegLowerSegment);
        rigidBodies.Remove(_bodyParts.LeftLegUpperSegment);
        rigidBodies.Remove(_bodyParts.RightLegLowerSegment);
        rigidBodies.Remove(_bodyParts.RightLegUpperSegment);
    }

    public void RepairBody()
    {
        _bodyParts.Body.Collided = false;
    }
}

public class BodyParts
{
    public Pole LeftLegUpperSegment;
    public Pole LeftLegLowerSegment;
    public Pole RightLegUpperSegment;
    public Pole RightLegLowerSegment;
    
    public Hull Body;
}