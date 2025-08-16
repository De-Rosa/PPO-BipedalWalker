## **NEA Project**
AQA A Level Computer Science project created for the NEA (Non-Exam Assessment).  
An environment is created using a rigid-body physics engine, where a bipedal walker attempts to walk across the screen.  
The ML model implemented is PPO (a reinforcement learning algorithm), based on my own matrix library, with Adam being used as the optimization algorithm.  

## Contents
`Bodies/` contains code for the physics engine (where `Materials/` defines materials used in the environment and `Objects/` contains templates for shapes and AABB logic).
`Walker/` contains code for the neural net and PPO setup alongside the generic updates for the bipedal walker.
`Input/` and `Rendering/` contain code for the user interface.
  
### **Credits:**
https://github.com/b2developer/SpidermanPPO - Introduction to PPO and the algorithm used for backwards propagation,  
https://docs.google.com/document/d/1FZZvz0JMHKWOOVlXnrmeRMoGpyjqa0m6Q0S2qLECDpA - Description of implementing PPO in a continuous environment,  
https://huggingface.co/blog/deep-rl-ppo#introducing-the-clipped-surrogate-objective - Explanation of PPO's clipped objective,  
https://arxiv.org/pdf/1707.06347 - The PPO paper,  
https://fse.studenttheses.ub.rug.nl/25709/1/mAI_2021_BickD.pdf - Used for the entire PPO differential calculation for back-propagation,  
& others referenced inside the code (where appropiate).  
