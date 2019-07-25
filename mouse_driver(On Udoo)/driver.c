#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/module.h>
#include <linux/kdev_t.h>
#include <linux/fs.h>
#include <linux/cdev.h>
#include <linux/device.h>
#include<linux/slab.h>                 //kmalloc()
#include<linux/uaccess.h>              //copy_to/from_user()
#include <linux/ioctl.h>

#include <linux/input.h>
#include <linux/usb.h> 
#include <linux/poll.h>
#include <linux/miscdevice.h>
#include <asm/uaccess.h>

#define DEVICE_NAME "mousek"
#define MAX_SCREEN 100
#define MIN_SCREEN 0
 
#define WR_VALUE _IOW('a','a',int32_t*)
#define RD_VALUE _IOR('a','b',int32_t*)
 
int32_t value = 0;
int Major;
typedef struct params
{
  int32_t dx,dy,env;
}params;

struct mousek_device {
    signed char data[4];     /* use a 4-byte protocol */
    struct urb urb;          /* USB Request block, to get USB data*/
    struct input_dev *idev;   /* input device, to push out input  data */
    int x, y;                /* keep track of the position of this device */
};

int pkt_drop_flag=0;   
dev_t dev = 0;
static struct mousek_device *mouse; 

int init_module(void);
void cleanup_module(void);
static int etx_open(struct inode *inode, struct file *file);
static int etx_release(struct inode *inode, struct file *file);
static ssize_t etx_read(struct file *filp, char __user *buf, size_t len,loff_t * off);
static ssize_t etx_write(struct file *filp, const char *buf, size_t len, loff_t * off);
static long etx_ioctl(struct file *file, unsigned int cmd, unsigned long arg);
 
static struct file_operations mousek_fops =
{
        .owner          = THIS_MODULE,
        .read           = etx_read,
        .write          = etx_write,
        .open           = etx_open,
        .unlocked_ioctl = etx_ioctl,
        .release        = etx_release,
};
 
static int etx_open(struct inode *inode, struct file *file)
{
        printk(KERN_INFO "Device File Opened...!!!\n");
        return 0;
}
 
static int etx_release(struct inode *inode, struct file *file)
{
        printk(KERN_INFO "Device File Closed...!!!\n");
        return 0;
}
 
static ssize_t etx_read(struct file *filp, char __user *buf, size_t len, loff_t *off)
{
        printk(KERN_INFO "Read Function\n");
        return 0;
}
static ssize_t etx_write(struct file *filp, const char __user *buf, size_t len, loff_t *off)
{
        printk(KERN_INFO "Write function\n");
        return 0;
}
 
static long etx_ioctl(struct file *file, unsigned int cmd, unsigned long arg)
{           
        int rc;
        struct params *obj= kmalloc(sizeof(struct params), GFP_DMA);
        struct input_dev *dev = mouse->idev;
        switch(cmd) 
        {
                case WR_VALUE:
                        mouse->data[1]=mouse->data[2]=mouse->data[3]=0;
                        rc = copy_from_user(obj, (void *)arg, sizeof(struct params));
                        printk(KERN_INFO "\ncopy_from_user() = %d.\n", rc);
                        printk(KERN_INFO "\ndx= %d dy= %d env= %d \n", obj->dx,obj->dy,obj->env);
                        mouse->data[1]=obj->dx;
                        mouse->data[2]=obj->dy;
                        mouse->data[3]=obj->env;
                        input_report_rel(dev, REL_X, mouse->data[1]);
                        input_sync(dev); 
                        input_report_rel(dev, REL_Y, mouse->data[2]);
                        input_sync(dev);   
                        //handle queue clicks
                        if(mouse->data[3] != 0){
                            if(mouse->data[3] == 1){
                                input_report_key(dev, BTN_LEFT, 1);                 
                            }else if(mouse->data[3] == 2){
                                input_report_key(dev, BTN_LEFT, 0);
                            }else if(mouse->data[3] == 3){
                                input_report_key(dev, BTN_RIGHT, 1);
                            }else if(mouse->data[3] == 4){
                                input_report_key(dev, BTN_RIGHT, 0);
                            }
                            input_sync(dev);
                        } 
                            
                        input_sync(dev);
                case RD_VALUE:
                        copy_to_user((void *)arg, obj, sizeof(value));
                        break;
        }
        return 0;
}
 
int init_module(void)
{
    int retval;
    
    Major = register_chrdev(0, DEVICE_NAME, &mousek_fops);
    if (Major < 0) {
      printk(KERN_ALERT "Registering char device failed with %d\n", Major);
      return Major;
    }
    
    //if (request_irq(BUTTON_IRQ, mousek_interrupt, 0, DEVICE_NAME, NULL)) {
    //  printk(KERN_ERR "mousek: Can't allocate irq \n");
    //    return -EBUSY;
    //}
    
struct input_dev *input_dev;

    /* allocate and zero a new data structure for the new device */
    mouse = kmalloc(sizeof(struct mousek_device), GFP_KERNEL);
    if (!mouse) return -ENOMEM; /* failure */
    memset(mouse, 0, sizeof(*mouse));

    input_dev = input_allocate_device();
    if (!input_dev) {
        printk(KERN_ERR "mousek.c: Not enough memory\n");
        retval = -ENOMEM;
        //goto err_free_irq;
    }
    //updating struct
    mouse->idev = input_dev;
    
    /* tell the features of this input device: fake only keys */
    //mouse->idev.evbit[0] = BIT(EV_KEY);
    /* and tell which keys: only the arrows */
    //set_bit(103, mouse->idev.keybit); /* Up    */
    //set_bit(105, mouse->idev.keybit); /* Left  */
    //set_bit(106, mouse->idev.keybit); /* Right */
    //set_bit(108, mouse->idev.keybit); /* Down  */
    
    input_dev->evbit[0] = BIT_MASK(EV_KEY) | BIT_MASK(EV_REL);
    //input_dev->evbit[0] = BIT_MASK(EV_KEY) | BIT_MASK(EV_ABS);
    
    //set_bit(103, input_dev->keybit); /* Up    */

    input_dev->keybit[BIT_WORD(BTN_MOUSE)] = BIT_MASK(BTN_LEFT) | BIT_MASK(BTN_RIGHT) | BIT_MASK(BTN_MIDDLE);
    input_dev->relbit[0] = BIT_MASK(REL_X) | BIT_MASK(REL_Y) | BIT_MASK(REL_WHEEL);
    //input_dev->absbit[0] = BIT_MASK(ABS_X) | BIT_MASK(ABS_Y);
    
    //input_set_abs_params(input_dev, ABS_X, MIN_SCREEN, MAX_SCREEN, 0, 0);
    //input_set_abs_params(input_dev, ABS_Y, MIN_SCREEN, MAX_SCREEN, 0, 0);

    
    input_dev->name = DEVICE_NAME;  
    input_set_drvdata(input_dev, mouse);
    
    retval = input_register_device(input_dev);
    if (retval) {
        printk(KERN_ERR "mousek: Failed to register device\n");
        goto err_free_dev;
    }

    
    printk(KERN_INFO "I was assigned major number %d. To talk to\n", Major);
    printk(KERN_INFO "the driver, create a dev file with\n");   
    printk(KERN_INFO "'mknod /dev/%s c %d 0'.\n", DEVICE_NAME, Major);
    printk(KERN_INFO "Try various minor numbers. Try to cat and echo to\n");
    printk(KERN_INFO "the device file.\n");
    printk(KERN_INFO "Remove the device file and module when done.\n");
    
    
return 0;

err_free_dev:
    input_free_device(mouse->idev);
    kfree(mouse);
//err_free_irq:
//  free_irq(BUTTON_IRQ, button_interrupt);
return retval;
}

void cleanup_module(void)
{
    /*
    * Unregister the device
    */
    if(!mouse) return;
    
    input_unregister_device(mouse->idev);
    kfree(mouse);   
    unregister_chrdev(Major, DEVICE_NAME);
    
    printk(KERN_ALERT "Uninstalled. Delete device from dev.");
    
    //ret = free_irq(BUTTON_IRQ, mousek_interrupt);
    //if (ret < 0)
    //  printk(KERN_ALERT "Error in freeing irq: %d\n", ret);

} 
MODULE_LICENSE("GPL");
MODULE_AUTHOR("CHM Team - LEC 2019");
MODULE_DESCRIPTION("A simple device driver");
MODULE_VERSION("1.0");