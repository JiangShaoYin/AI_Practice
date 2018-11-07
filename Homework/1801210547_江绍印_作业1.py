class Vehicle(object):
	def Move(self):
		print ("Vehicle is moving")
	def Transpot(self):
		print ("Vehicle is transpoting")

class Car(Vehicle):
	def __init__(self,weight,height):
		self.weight = weight
		self.height = height
	def SpeedUp(self):
		print ("Car is speeding up")
	def SpeedDown(self):
		print("Car is speeding down")
	def ShowInfo(self):
		print ("weight is %d,height is %d"%(self.weight,self.height))

Benz = Car(2000,1100)
Benz.SpeedUp()
Benz.ShowInfo()
Benz.Move()
Benz.Transpot()
