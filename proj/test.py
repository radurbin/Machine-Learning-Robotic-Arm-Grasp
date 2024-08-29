#add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)
import pybullet as p
from graspTypes import graspTypes
#from pybullet_envs.bullet.kukaGymEnv import KukaGymEnv
from sawyerEnv import sawyerEnv
import time

def main():

  #grasp types = ["poPmAb25", "poPmAd25", "poPdAb2", "poPdAb23", "poPdAb24", "pPdAb2", "pPdAb23", "pPdAb24", "pPdAb25", "pPdAd25"] 
  graspType = "poPdAb23"
  orientation = 2 # 0 from side, 1 from above, 2 from above 1

  environment = sawyerEnv(renders=True, isDiscrete=False, maxSteps=10000000, graspType = graspType, orientation = orientation)
  readings = [0] * 35
  motorsIds = []

  dv = 0.01
  motorsIds.append(environment._p.addUserDebugParameter("x", -0.1, 0.1, 0))
  motorsIds.append(environment._p.addUserDebugParameter("y", -0.1, 0.1, 0))
  motorsIds.append(environment._p.addUserDebugParameter("z", -0.1, 0.1, 0))
  motorsIds.append(environment._p.addUserDebugParameter("thumb", 0, 1, 0))
  motorsIds.append(environment._p.addUserDebugParameter("index", 0, 1, 0))
  motorsIds.append(environment._p.addUserDebugParameter("middle", 0, 1, 0))
  motorsIds.append(environment._p.addUserDebugParameter("ring", 0, 1, 0))
  motorsIds.append(environment._p.addUserDebugParameter("pinky", 0, 1, 0))
  done = False
  action = []
  while (not done):

    action = []
    for motorId in motorsIds:
      action.append(environment._p.readUserDebugParameter(motorId))

    environment.step(action)

    qKey = ord('q')
    keys = p.getKeyboardEvents()
    if qKey in keys and keys[qKey]&p.KEY_WAS_TRIGGERED:
      environment.close()
      break;

  environment.close()

if __name__ == "__main__":
  main()


