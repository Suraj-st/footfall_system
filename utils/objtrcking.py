class ObjTracker:
	def __init__(self, objectID, cp):
		# store the object ID, then initialize a list of cps
		# using the current cp
		self.objectID = objectID
		self.cps = [cp]

		# initialize a boolean used to indicate if the object has
		# already been counted or not
		self.counted = False