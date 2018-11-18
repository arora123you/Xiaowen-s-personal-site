import csv
filename='trum.csv'
with open(filename) as f:
    reader = csv.reader(f)
    a=list(reader)
    #print(a_ll)
    tmp=[[a[1][1],eval(a[1][3])]]
    new=[]
    for i in range(2,len(a)):
    	#print(a[i-1][1][-2:])
    	t1=eval(tmp[-1][0][-2])*10+eval(tmp[-1][0][-1])+eval(tmp[-1][0][-5])*600+eval(tmp[-1][0][-4])*60
    	t2=eval(a[i][1][-2])*10+eval(a[i][1][-1])+eval(a[i][1][-5])*600+eval(a[i][1][-4])*60
    	print(t1,t2)
    	if (t1 == t2 ) or (t1 == t2 - 1):
    		tmp.append([tmp[-1][0],eval(a[i][3])])
    	else:
    		new.append(tmp)
    		tmp=[[a[i][1],eval(a[i][3])]]

res=[];ori=[]
for i in range(len(new)):
	tmp=new[i][0][1];a1=0;a2=0;a3=0;a4=0;a5=0;a6=0
	for j in range(len(new[i])):
		a1+=new[i][j][1][0]
		a2+=new[i][j][1][1]
		a3+=new[i][j][1][2]
		a4+=new[i][j][1][3]
		a5+=new[i][j][1][4]
		a6+=new[i][j][1][5]
	ori.append([new[i][0][0],a1,a2,a3,a4,a5,a6])
	a1=float(a1)/len(new[i])
	a2=float(a2)/len(new[i])
	a3=float(a3)/len(new[i])
	a4=float(a4)/len(new[i])
	a5=float(a5)/len(new[i])
	a6=float(a6)/len(new[i])
	res.append([new[i][0][0],a1,a2,a3,a4,a5,a6])


with open("tt.csv","w") as csvfile: 
    writer = csv.writer(csvfile)
    writer.writerows(res)
