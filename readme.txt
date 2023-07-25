There are two implementations: 
1. for binary images
2. for gray images

In folders binary_images and gray_images there are binary and gray images respectively of sizes 1024x1024, 2048x2048, 4096x4096 and 8192x8192 bytes.

The Div2K dataset can be found online.

Execution example:
make clean
make
./pibr /nfsshare/images/binary/shapes1024.raw 1024 1024 64