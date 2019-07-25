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
int rc, fd,i ;
typedef struct params
{
  int32_t dx,dy,env;
  
}params;
struct params data;
int open_file(void)
{

        fd = open("/dev/mousek", O_RDWR);
        if(fd < 0) {
                printf("Cannot open device file...\n");
                return 0;
        }
return fd;	
}
void close_file(void)
{
	close(fd);
	exit(0);
	
}
void write_value(int32_t dx,int32_t dy,int32_t env)
{
	data.dx = dx;
	data.dy = dy;
	data.env = env;
	rc = ioctl(fd, WR_VALUE, &data);
        printf("ioctl(fd, WR_VALUE, move) = %d\n", rc);
}
int main()
{       
}

