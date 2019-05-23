#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include<sys/ioctl.h>
 
#define WR_VALUE _IOW('a','a',int32_t*)
#define RD_VALUE _IOR('a','b',int32_t*)

typedef struct params
{
  int32_t dx,dy,env;
  
}params;
 
int main()
{
        int rc, fd,i ;
        struct params data;
        data.dx = data.dy = data.env =0;

        fd = open("/dev/mousek", O_RDWR);
        if(fd < 0) {
                printf("Cannot open device file...\n");
                return 0;
        }
        while(1) {
        switch(rand() % 8) {
        case 0:
            data.dx = -1;
            data.dy = 1;
            break;
        case 1:
            data.dx = 1;
            data.dy = 1;
            break;
        case 2:
            data.dx = 1;
            data.dy = -1;
            break;
        case 3:
            data.dx = -1;
            data.dy = -1;
            break;
        case 4:
            data.dx = 1;
            data.dy = 0;
            break;
        case 5:
            data.dx = -1;
            data.dy = 0;
            break;
        case 6:
            data.dx = 0;
            data.dy = 1;
            break;
        case 7:
            data.dx = 0;
            data.dy = -1;
            break;            
        }
            for (i=0;i<rand()%100;i++){
            rc = ioctl(fd, WR_VALUE, &data);
            printf("ioctl(fd, WR_VALUE, move) = %d\n", rc);
            usleep(15000);}
       }
        close(fd);

        exit(0);

}

