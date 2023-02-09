import numpy as np

partitions = np.loadtxt("partitions_0.txt", dtype=int)
centroids = np.loadtxt("centroids_0.txt", dtype=int)

def encode(val):
    centroid = 0
    for i, partition in enumerate(partitions):
        if val in partition:
            centroid = i
            break
    return centroids[centroid]

test = open("./org/test.pgm", 'wb')
f = open("./org/98.pgm", 'rb')
    
line = f.readline()
    # Check header is P5
if line != str.encode('P5\n'):
    f.close() 
    print("error1")
test.write(line)

line = f.readline()
test.write(line)

line = f.readline()
test.write(line)
(width, height) = [int(i) for i in line.split()] 

line = f.readline()
if line != str.encode('255\n'):
    f.close()
    print("error2")
test.write(line)

for y in range(height):
    for x in range(width):
        pixel_val = ord(f.read(1))
        print(encode(pixel_val))
        test.write(str.encode(chr(encode(pixel_val))))

test.close()
f.close()



