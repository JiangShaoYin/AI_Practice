
f = open('./homework.txt','w')

for column in range(1,10):
	for row in range(1,10):
		if row<=column:
			f.write('%d*%d=%d	'%(column,row,column*row))
	f.write('\n')
f.close()

		
