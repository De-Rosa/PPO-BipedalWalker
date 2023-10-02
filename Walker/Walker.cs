using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.Xna.Framework;
using Physics.Bodies;
using Physics.Materials;
using Physics.Objects.RigidBodies;
using Physics.Rendering;
using Physics.Walker.PPO;
using Matrix = Physics.Walker.PPO.Matrix;

namespace Physics.Walker;

// Walker class, holds the body parts to the walker and the "brain" AI.
// Provides methods for taking actions and receiving a state.
public class Walker
{
    private readonly IMaterial _material;
    private readonly List<Joint> _joints;
    private Vector2 _position;
    private Vector2 _previousPosition;
    private BodyParts _bodyParts;
    private readonly PPOAgent _brain;

    public bool Terminal;

    public Walker()
    {
        _joints = new List<Joint>();
        _brain = new PPOAgent(8, 4);
        _bodyParts = new BodyParts();
        _material = new Carpet();
        _position = new Vector2(125, 800);
        _previousPosition = _position;
        Terminal = false;
    }
    
    // Creates the walker's physical body.
    // Instantiates the rigid body and connects them with joints.
    // Applies gravity which is a constant acceleration of 980 (positive since y increases as it 
    // goes down the screen)
    public void CreateCreature(List<RigidBody> rigidBodies)
    {
        CreateBodies(rigidBodies);
        CreateJoints();
        AddAssociatedBodies();
        AddAcceleration(new Vector2(0, 980));
    }

    // Walker update function, updates its position based on the walker's midpoint.
    public void Update()
    {
        _previousPosition = _position;
        _position = _bodyParts.Body.GetCentroid();
        if (_bodyParts.Body.Collided) Terminal = true;
    }

    // Samples action from the PPO agent, based on a state given by the environment.
    // Outputs the action and it's log probability, to be stored in the trajectory information.
    public Matrix GetActions(Matrix state, out Matrix logProbabilities)
    {
        Matrix actions = _brain.SampleActions(state, out logProbabilities, out Matrix mean, out Matrix std);
        return actions;
    }

    // Inputs an action and applies the value set to each joint.
    // The torque's range is -1 ≤ x ≤ 1, as it is clipped by the environment.
    public void TakeActions(Matrix actions)
    {
        if (actions.GetHeight() != _joints.Count) return;
        
        for (int i = 0; i < _joints.Count; i++)
        {
            float torque = actions.GetValue(i, 0);
            _joints[i].SetTorque(torque);
        }
    }

    // Trains the PPO agent based on a trajectory.
    // Renderer is passed through in order to render the training information.
    public void Train(Trajectory trajectory, Renderer renderer)
    {
        _brain.Train(trajectory, renderer);
    }

    // Returns the colors of each joint based on their torque.
    // If the joint has torque 1, it will be red. If has torque -1, it will be blue.
    public List<Tuple<Vector2, Color>> GetJointColors()
    {
        List<Tuple<Vector2, Color>> colors = new List<Tuple<Vector2, Color>>();
        
        foreach (var joint in _joints)
        {
            Vector2 position = (joint.GetPointA() + joint.GetPointB()) / 2f;
            float value = (1f + joint.GetTorque()) / 2f;
            Color color = new Color(value * 255, 0, (1 - value) * 255, 100);
            colors.Add(new Tuple<Vector2, Color>(position, color));
        }

        return colors;
    }

    // Returns the joints list.
    public List<Joint> GetJoints()
    {
        return _joints;
    }

    // Returns the current position of the walker.
    public Vector2 GetPosition()
    {
        return _position;
    }
    
    // Returns the previous change in position for use in reward calculation.
    public Vector2 GetChangeInPosition()
    {
        return _position - _previousPosition;
    }

    // Returns a state matrix involving the walker's current values.
    // 
    public Matrix GetState()
    {
        float[] values = new[]
        {
            _bodyParts.Body.GetAngle(),
            _bodyParts.Body.GetAngularVelocity(),
            
            _bodyParts.Body.GetLinearVelocity().X / Game1.FrameRate,
            _bodyParts.Body.GetLinearVelocity().Y / Game1.FrameRate,
            
            _bodyParts.LeftLegLowerSegment.GetAngle(),
            _bodyParts.LeftLegUpperSegment.GetAngle(),
            _bodyParts.RightLegLowerSegment.GetAngle(),
            _bodyParts.RightLegUpperSegment.GetAngle()
        };

        return Matrix.FromValues(values);
    }

    // Creates the rigid bodies for each of the body parts of the walker.
    private void CreateBodies(List<RigidBody> rigidBodies)
    {
        Skeleton bodySkeleton = new Skeleton();
        bodySkeleton.AddVectors(new Vector2[]
        {
            new(_position.X + 20, _position.Y + 20), // bottom right
            new(_position.X, _position.Y + 20), // bottom middle
            new(_position.X - 20, _position.Y + 20), // bottom left
            new(_position.X - 20, _position.Y - 20), // top left
            new(_position.X + 20, _position.Y - 20) // top right
        });

        _bodyParts.Body = Hull.FromSkeleton(_material, bodySkeleton);
        _bodyParts.Body.SetInverseInertia(0.0005f);

        _bodyParts.LeftLegUpperSegment = Pole.FromSize(_material, _position + new Vector2(0, 20), 75);
        _bodyParts.LeftLegLowerSegment = Pole.FromSize(_material, _position + new Vector2(0, 50f), 75);

        _bodyParts.RightLegUpperSegment = Pole.FromSize(_material, _position + new Vector2(0, 20), 75);
        _bodyParts.RightLegLowerSegment = Pole.FromSize(_material, _position + new Vector2(0, 50f), 75);

        rigidBodies.AddRange(new RigidBody[] {_bodyParts.LeftLegLowerSegment, _bodyParts.LeftLegUpperSegment, _bodyParts.Body, _bodyParts.RightLegLowerSegment, _bodyParts.RightLegUpperSegment});
    }

    // Creates the joints between each legs of the walker.
    private void CreateJoints()
    {
        Joint bodyJointLeft = new Joint(_bodyParts.Body, _bodyParts.LeftLegUpperSegment, 1, 4);
        Joint bodyJointRight = new Joint(_bodyParts.Body, _bodyParts.RightLegUpperSegment, 1, 4);
        Joint leftJoint = new Joint(_bodyParts.LeftLegUpperSegment, _bodyParts.LeftLegLowerSegment, 2, 3);
        Joint rightJoint = new Joint(_bodyParts.RightLegUpperSegment, _bodyParts.RightLegLowerSegment, 2, 3);

        _joints.AddRange(new []{bodyJointLeft, bodyJointRight,  leftJoint, rightJoint});
    }

    // Adds acceleration to each of the rigid bodies, mainly used to add gravity to all of the body parts.
    private void AddAcceleration(Vector2 acceleration)
    {
        _bodyParts.LeftLegUpperSegment.AddAcceleration(acceleration);
        _bodyParts.LeftLegLowerSegment.AddAcceleration(acceleration);
        _bodyParts.RightLegUpperSegment.AddAcceleration(acceleration);
        _bodyParts.RightLegLowerSegment.AddAcceleration(acceleration);
        _bodyParts.Body.AddAcceleration(acceleration);
    }

    // Associated bodies are bodies which do not resolve or detect collisions between each other.
    // This function makes it so that the walker doesn't collide with its legs, etc.
    private void AddAssociatedBodies()
    {
        _bodyParts.LeftLegUpperSegment.AddAssociatedBodies(new RigidBody[] {_bodyParts.RightLegUpperSegment, _bodyParts.RightLegLowerSegment, _bodyParts.Body});
        _bodyParts.LeftLegLowerSegment.AddAssociatedBodies(new RigidBody[] {_bodyParts.RightLegUpperSegment, _bodyParts.RightLegLowerSegment, _bodyParts.Body});
        _bodyParts.RightLegUpperSegment.AddAssociatedBodies(new RigidBody[] {_bodyParts.LeftLegUpperSegment, _bodyParts.LeftLegLowerSegment, _bodyParts.Body});
        _bodyParts.RightLegLowerSegment.AddAssociatedBodies(new RigidBody[] {_bodyParts.LeftLegUpperSegment, _bodyParts.LeftLegLowerSegment, _bodyParts.Body});
        _bodyParts.Body.AddAssociatedBodies(new RigidBody[] {_bodyParts.LeftLegUpperSegment, _bodyParts.RightLegUpperSegment, _bodyParts.LeftLegLowerSegment, _bodyParts.RightLegLowerSegment});
    }

    // Resets the walker's physical state and variables that are associated with them.
    public void Reset(List<RigidBody> rigidBodies)
    {
        RemoveRigidObjects(rigidBodies);        
        _joints.Clear();
        _bodyParts = new BodyParts();
        Terminal = false;
        
        _position = new Vector2(125, 800);
        _previousPosition = _position;
        
        CreateCreature(rigidBodies);
    }

    // Removes the walker's body from the rigid bodies list.
    private void RemoveRigidObjects(List<RigidBody> rigidBodies)
    {
        if (rigidBodies.Count < 6) return;
        rigidBodies.Remove(_bodyParts.Body);
        rigidBodies.Remove(_bodyParts.LeftLegLowerSegment);
        rigidBodies.Remove(_bodyParts.LeftLegUpperSegment);
        rigidBodies.Remove(_bodyParts.RightLegLowerSegment);
        rigidBodies.Remove(_bodyParts.RightLegUpperSegment);
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