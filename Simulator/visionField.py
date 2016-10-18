from math import *

def rad2deg(ang):
	return 180.0 * ang / pi

def norm(v):
	return sqrt(v[0]*v[0] + v[1]*v[1])

def dot(a, b):
	return a[0]*b[0] + a[1]*b[1]

def dotproduct(a, b):
	return  acos( dot(a,b) / (norm(a)*norm(b))  )     

def angleBetween(a,b):
    return rad2deg(dotproduct(a,b))

def inVisionCone(enemyVector, leftVector, rightVector, visionFieldAngle):
    if (angleBetween(leftVector, enemyVector) <= visionFieldAngle) and (angleBetween(enemyVector, rightVector) <= visionFieldAngle):
        return True
    else:
        return False

if __name__ == "__main__":

    robotCenter            = [ 0.0, 0.0]
    enemyCenter            = [ 0.0, 3.0]
    endPointLeftVisionRay  = [-1.0, 1.0]
    endPointRightVisionRay = [ 1.0, 1.0]

    leftVector  = [endPointLeftVisionRay[0]  - robotCenter[0], endPointLeftVisionRay[1]  - robotCenter[1]]
    rightVector = [endPointRightVisionRay[0] - robotCenter[0], endPointRightVisionRay[1] - robotCenter[1]]
    enemyVector = [enemyCenter[0]            - robotCenter[0], enemyCenter[1]            - robotCenter[1]]

    visionFieldAngle = angleBetween(leftVector,  rightVector) # or hardwired
    
    print "Angle leftVisionRay  <--> rightVisionRay\t%.2f" % angleBetween(leftVector,  rightVector)
    print "Angle leftVisionRay  <--> enemy\t\t\t%.2f" % angleBetween(leftVector,  enemyVector)
    print "Angle rightVisionRay <--> enemy\t\t\t%.2f" % angleBetween(rightVector, enemyVector)
    print "Within vision cone? %s"    % inVisionCone(enemyVector, leftVector, rightVector, visionFieldAngle)
