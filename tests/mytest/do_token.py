f=open("tokens","r")
content=f.read().split("\n")
string1=""
string2=""
for i in range(len(content)):
    if i%2==0:
        string1+=str(content[i].split(",")[-1])+"\n"
    else:
        string2+=str(content[i].split(",")[-1])+"\n"

print(string1)

print("!!!!!!!!")
print(string2)

for i in range(1,int(len(content)/2)+1):
    print(i*512)

    

