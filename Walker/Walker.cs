using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.Xna.Framework;
using Physics.Bodies;
using Physics.Materials;
using Physics.Objects;
using Physics.Objects.RigidBodies;
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
    private readonly PPO.PPOAgent _brain;

    public bool Terminal;

    public Walker()
    {
        _joints = new List<Joint>();
        _brain = new PPO.PPOAgent(17, 8);
        _bodyParts = new BodyParts();
        _material = new Carpet();
        _position = new Vector2(125, 800);
        _previousPosition = _position;
        Terminal = false;
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
        _position = _bodyParts.Body.GetCentroid();
        if (_bodyParts.Body.Collided) Terminal = true;
    }

    public Matrix GetActions(Matrix state, out Matrix probabilities, out Matrix mean, out Matrix std)
    {
        return _brain.SampleActions(state, out probabilities, out mean, out std);
    }

    public void TakeActions(Matrix actions)
    {
        for (int i = 2; i < _joints.Count; i++)
        {
            float torque = actions.GetValue(i, 0);
            _joints[i].SetTorque(torque);
        }
    }

    public void Train(Trajectory trajectory)
    {
        _brain.Train(trajectory);
    }

    public void Save(string criticFileLocation, string actorFileLocation)
    {
        _brain.Save(criticFileLocation, actorFileLocation);
    }

    public void Load(string criticFileLocation, string actorFileLocation)
    {
        _brain.Load(criticFileLocation, actorFileLocation);
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
    
    public Vector2 GetChangeInPosition()
    {
        return _position - _previousPosition;
    }

    public Matrix GetState()
    {
        float[] values = new[]
        {
            _bodyParts.LeftLegLowerSegment.GetAngularVelocity(),
            _bodyParts.LeftLegUpperSegment.GetAngularVelocity(),
            _bodyParts.RightLegLowerSegment.GetAngularVelocity(),
            _bodyParts.RightLegUpperSegment.GetAngularVelocity(), 
            _bodyParts.Body.GetAngularVelocity(),
                
            _bodyParts.LeftLegLowerSegment.GetLinearVelocity().Y,
            _bodyParts.LeftLegUpperSegment.GetLinearVelocity().Y,
            _bodyParts.RightLegLowerSegment.GetLinearVelocity().Y,
            _bodyParts.RightLegUpperSegment.GetLinearVelocity().Y, 
            _bodyParts.Body.GetLinearVelocity().Y,
                
            _bodyParts.LeftLegLowerSegment.GetLinearVelocity().X,
            _bodyParts.LeftLegUpperSegment.GetLinearVelocity().X,
            _bodyParts.RightLegLowerSegment.GetLinearVelocity().X,
            _bodyParts.RightLegUpperSegment.GetLinearVelocity().X, 
            _bodyParts.Body.GetLinearVelocity().X,
            
            _bodyParts.LeftLegLowerSegment.Collided ? 1 : 0,
            _bodyParts.RightLegLowerSegment.Collided ? 1 : 0
        };
        
        return Matrix.FromValues(values);
    }

    private void CreateBodies(List<IObject> rigidBodies)
    {
        Vector2[] squareVectors = new Vector2[]
        {
            new Vector2(_position.X + 7.5f, _position.Y + 5f - 10), 
            new Vector2(_position.X, _position.Y + 5f - 10),
            new Vector2(_position.X - 7.5f, _position.Y + 5f - 10),
            new Vector2(_position.X - 7.5f, _position.Y - 5f - 10),
            new Vector2(_position.X, _position.Y - 5f - 10),
            new Vector2(_position.X + 7.5f, _position.Y - 5f - 10)
        };
        
        Skeleton squareSkeleton = new Skeleton();
        squareSkeleton.AddVectors(squareVectors);

        Skeleton squareSkeleton2 = new Skeleton();
        squareSkeleton2.AddVectors(squareVectors);
        
        Skeleton bodySkeleton = new Skeleton();
        bodySkeleton.AddVectors(new Vector2[]
        {
            new Vector2(_position.X + 20, _position.Y + 20 - 50), // bottom right
            new Vector2(_position.X, _position.Y + 20 - 50), // bottom middle
            new Vector2(_position.X - 20, _position.Y + 20 - 50), // bottom left
            new Vector2(_position.X - 20, _position.Y - 20 - 50), // top left
            new Vector2(_position.X + 20, _position.Y - 20 - 50) // top right
        });

        _bodyParts.Body = Hull.FromSkeleton(_material, bodySkeleton, isFragile: true);
        _bodyParts.Body.SetInverseInertia(0.001f);

        _bodyParts.LeftSquare = Hull.FromSkeleton(_material, squareSkeleton);
        _bodyParts.RightSquare = Hull.FromSkeleton(_material, squareSkeleton2);

        _bodyParts.LeftLegUpperSegment = Pole.FromSize(_material, _position + new Vector2(0, 20), 75);
        _bodyParts.LeftLegLowerSegment = Pole.FromSize(_material, _position + new Vector2(0, 50), 75, isFloor: true);

        _bodyParts.RightLegUpperSegment = Pole.FromSize(_material, _position + new Vector2(0, 20), 75);
        _bodyParts.RightLegLowerSegment = Pole.FromSize(_material, _position + new Vector2(0, 50), 75, isFloor: true);

        rigidBodies.AddRange(new IObject[] {_bodyParts.LeftSquare, _bodyParts.RightSquare, _bodyParts.LeftLegLowerSegment, _bodyParts.LeftLegUpperSegment, _bodyParts.Body, _bodyParts.RightLegLowerSegment, _bodyParts.RightLegUpperSegment});
    }

    private void CreateJoints()
    {
        Joint squareLeftJoint = new Joint(_bodyParts.Body, _bodyParts.LeftSquare, 1, 4);
        Joint squareRightJoint = new Joint(_bodyParts.Body, _bodyParts.RightSquare, 1, 4);
        Joint bodyJointLeft = new Joint(_bodyParts.LeftSquare, _bodyParts.LeftLegUpperSegment, 1, 4);
        Joint bodyJointRight = new Joint(_bodyParts.RightSquare, _bodyParts.RightLegUpperSegment, 1, 4);
        Joint leftJoint = new Joint(_bodyParts.LeftLegUpperSegment, _bodyParts.LeftLegLowerSegment, 2, 3);
        Joint rightJoint = new Joint(_bodyParts.RightLegUpperSegment, _bodyParts.RightLegLowerSegment, 2, 3);

        _joints.AddRange(new []{squareLeftJoint, squareRightJoint, bodyJointLeft, bodyJointRight, leftJoint, rightJoint});
    }

    private void AddAcceleration(Vector2 acceleration)
    {
        _bodyParts.LeftLegUpperSegment.AddAcceleration(acceleration);
        _bodyParts.LeftLegLowerSegment.AddAcceleration(acceleration);
        _bodyParts.RightLegUpperSegment.AddAcceleration(acceleration);
        _bodyParts.RightLegLowerSegment.AddAcceleration(acceleration);
        _bodyParts.LeftSquare.AddAcceleration(acceleration);
        _bodyParts.RightSquare.AddAcceleration(acceleration);
        _bodyParts.Body.AddAcceleration(acceleration);
    }

    private void AddAssociatedBodies()
    {
        _bodyParts.LeftLegUpperSegment.AddAssociatedBodies(new RigidBody[] {_bodyParts.RightSquare, _bodyParts.LeftSquare, _bodyParts.RightLegUpperSegment, _bodyParts.RightLegLowerSegment});
        _bodyParts.LeftLegLowerSegment.AddAssociatedBodies(new RigidBody[] {_bodyParts.RightLegUpperSegment, _bodyParts.RightLegLowerSegment});
        _bodyParts.RightLegUpperSegment.AddAssociatedBodies(new RigidBody[] {_bodyParts.RightSquare, _bodyParts.LeftSquare, _bodyParts.LeftLegUpperSegment, _bodyParts.LeftLegLowerSegment});
        _bodyParts.RightLegLowerSegment.AddAssociatedBodies(new RigidBody[] {_bodyParts.LeftLegUpperSegment, _bodyParts.LeftLegLowerSegment});
        _bodyParts.LeftSquare.AddAssociatedBodies(new RigidBody[] {_bodyParts.Body, _bodyParts.RightLegUpperSegment, _bodyParts.RightLegLowerSegment, _bodyParts.RightSquare});
        _bodyParts.RightSquare.AddAssociatedBodies(new RigidBody[] {_bodyParts.Body, _bodyParts.LeftLegUpperSegment, _bodyParts.LeftLegLowerSegment, _bodyParts.LeftSquare});
        _bodyParts.Body.AddAssociatedBodies(new RigidBody[] {_bodyParts.LeftSquare, _bodyParts.RightSquare});
    }

    public void Reset()
    {
        _joints.Clear();
        _bodyParts = new BodyParts();
        Terminal = false;
        
        _position = new Vector2(125, 800);
        _previousPosition = _position;
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
    public Hull LeftSquare;
    public Hull RightSquare;
}