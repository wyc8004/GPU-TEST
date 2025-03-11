import pyopencl as cl
import numpy as np
import time
import tkinter
import pygame
import os
import atexit
from tkinter import messagebox as mb
SQUARE_SIZE = 32        
BACKGROUND_COLOR = (0, 0, 0) 
GREEN = (0, 255, 0)     
RED = (255, 0, 0)           
SCROLL_SPEED = 50
pygame.init()
@atexit.register
def onexit():
    from tkinter import messagebox as mb
    import tkinter
    import os
    rt=tkinter.Tk()
    rt.geometry('+50000+50000')
    mb.showerror('系统出错','请联系开发者(手机号18100182989)或尝试重新启动或重新安装以及检查你的硬件及驱动程序。')
    rt.destroy()
    os._exit(0)
def _gmt(gpu,big,limit):
    THRESHOLD_MS = limit
    pygame.init()
    infoObject = pygame.display.Info()
    screen_width, screen_height = infoObject.current_w, infoObject.current_h
    screen = pygame.display.set_mode((screen_width, screen_height), pygame.FULLSCREEN)
    screen.fill(BACKGROUND_COLOR)
    pygame.display.flip()
    ctx = cl.Context([gpu])
    queue = cl.CommandQueue(ctx)
    block_size_mb = 4                   # 4MB
    block_size = block_size_mb * 1024 * 1024  # 4MB 转换为字节
    total_size = big * block_size       # 2GB
    iterations = big                   # 512 个数据块
    mf = cl.mem_flags
    buffer_gpu = cl.Buffer(ctx, mf.WRITE_ONLY, total_size)
    data = np.random.randint(0, 256, size=block_size, dtype=np.uint8)
    data_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data)
    start_time = time.time()
    xx=0
    yy=0
    for i in range(iterations):
        start = time.time()
        cl.enqueue_copy(queue, buffer_gpu, data_gpu, src_offset=0, dst_offset=i * block_size).wait()
        end = time.time()
        elapsed = (end - start) * 1000  
        x=xx
        y=yy
        if elapsed > THRESHOLD_MS:
            color = RED
        else:
            color = GREEN
        pygame.draw.rect(screen, color, (x, y, SQUARE_SIZE, SQUARE_SIZE))
        xx+=32
        if xx>=screen_width:
            yy+=32
            xx=0
        pygame.display.update()
    end_time = time.time()
    elapsed_total = end_time - start_time
    pygame.quit()
    del buffer_gpu
    del ctx
def gmt(gpu,big,limit):
    try:
        _gmt(gpu,big,limit)
    except:
        try:
            pygame.quit()
        except:
            pass
def list_gpus():
    platforms = cl.get_platforms()
    gpus = []
    for platform in platforms:
        gpu_devices = platform.get_devices(device_type=cl.device_type.GPU)
        for gpu in gpu_devices:
            gpus.append(gpu)
    return gpus
def benchmark_matrix_multiplication(gpu, matrix_size):
    if not gpu:
        return None
    ctx = cl.Context([gpu])
    queue = cl.CommandQueue(ctx)
    kernel_code = """
    __kernel void matrix_multiply(__global const float *A, __global const float *B, __global float *C, const int N) {
        int row = get_global_id(0);
        int col = get_global_id(1);
        float value = 0.0f;
        for (int k = 0; k < N; k++) {
            value += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = value;
    }
    """
    program = cl.Program(ctx, kernel_code).build()
    A = np.random.rand(matrix_size, matrix_size).astype(np.float32)
    B = np.random.rand(matrix_size, matrix_size).astype(np.float32)
    C = np.empty_like(A)
    mf = cl.mem_flags
    A_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
    B_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
    C_buf = cl.Buffer(ctx, mf.WRITE_ONLY, C.nbytes)
    start_time = time.time()
    program.matrix_multiply(queue, C.shape, None, A_buf, B_buf, C_buf, np.int32(matrix_size))
    queue.finish()
    end_time = time.time()
    elapsed_time = end_time - start_time
    return elapsed_time
def test_gpu(matrix_size,gpuA):
    elapsed_time = benchmark_matrix_multiplication(gpuA, matrix_size)
    if elapsed_time is not None:
        return (f"矩阵乘法耗时: {elapsed_time:.10f} 秒",elapsed_time,matrix_size)
def writef(data,name):
    t=str(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
    n='{}-{}.txt'.format(t,name)
    f=open(os.getcwd()+os.sep+n,'w')
    f.write(str(data))
    f.close()
    return n
class GPUs(object):
    def __init__(self,y,obj):
        self.y=y
        self.name=obj.name
        self.obj=obj
        self.lb=None
        self.btn1=None
        self.btn2=None
        self.btn3=None
    def destroy(self,obja=None):
        try:
            obja.destroy()
        except:
            pass
    def update(self):
        self.destroy(self.lb)
        self.destroy(self.btn1)
        self.destroy(self.btn2)
        self.destroy(self.btn3)
        self.lb=tkinter.Label(rt,text='显卡:{}'.format(self.name))
        self.btn1=tkinter.Button(rt,text='跑分',command=self.pf)
        self.btn2=tkinter.Button(rt,text='烤鸡',command=self.bh)
        self.btn3=tkinter.Button(rt,text='显存测试',command=self.vramtest)
        self.lb.place(x=0,y=self.y)
        self.btn1.place(x=200,y=self.y)
        self.btn2.place(x=260,y=self.y)
        self.btn3.place(x=300,y=self.y)
    def pf(self):
        mb.showwarning('提示','软件可能卡死一段时间,请您放心,耐心等待。')
        test=tkinter.Tk()
        test.geometry('500x500')
        test.title('显卡 {} 跑分中'.format(self.name))
        yyy=0
        text={'gpu':self.name,'res':{}}
        for i in [256,512,1024]:
            tkinter.Label(test,text='{}数据块开始'.format(i)).place(x=0,y=yyy)
            ji=test_gpu(i,self.obj)
            tkinter.Label(test,text='{}数据块完成,结果:{}'.format(i,round(i/ji[1],4))).place(x=0,y=yyy)
            yyy+=50
            text['res']['{}'.format(i)]=[ji[1],round(i/ji[1],4)]
            test.update()
            rt.update()
        mb.showinfo('跑分完成','窗口即将关闭,请您放心,您的结果已经保存')
        test.destroy()
        os.startfile(writef(str(text),self.name))
    def bh(self):
        test=tkinter.Tk()
        test.protocol('WM_DELETE_WINDOW',self.end)
        test.geometry('500x500')
        test.title('显卡 {} 烤鸡中,FPS={}'.format(self.name,0))
        tkinter.Label(test,text='关闭本窗口结束烤鸡').place(x=0,y=0)
        self.bhing=True
        fps=0
        stat=round(time.time(),2)
        self.ttek=test
        ctx = cl.Context([self.obj])
        queue = cl.CommandQueue(ctx)
        kernel_code = """
        __kernel void matrix_multiply(__global const float *A, __global const float *B, __global float *C, const int N) {
            int row = get_global_id(0);
            int col = get_global_id(1);
            float value = 0.0f;
            for (int k = 0; k < N; k++) {
                value += A[row * N + k] * B[k * N + col];
            }
            C[row * N + col] = value;
        }
        """
        matrix_size=1024
        program = cl.Program(ctx, kernel_code).build()
        A = np.random.rand(matrix_size, matrix_size).astype(np.float32)
        B = np.random.rand(matrix_size, matrix_size).astype(np.float32)
        C = np.empty_like(A)
        mf = cl.mem_flags
        A_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
        B_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
        C_buf = cl.Buffer(ctx, mf.WRITE_ONLY, C.nbytes)
        while self.bhing:
            time.sleep(0.1)
            try:
                fps+=1
                program.matrix_multiply(queue, C.shape, None, A_buf, B_buf, C_buf, np.int32(matrix_size))
                if (round(time.time(),2)-stat) > 1 :
                    stat=round(time.time(),2)
                    test.title('显卡 {} 烤鸡中,FPS={}'.format(self.name,fps))
                    fps=0
                rt.update()
                test.update()
            except:
                break
        queue.finish()
    def end(self):
        mb.showwarning('提示','软件可能卡死一段时间,请您放心,耐心等待。')
        self.ttek.destroy()
        self.bhing=False
    def vramtest(self):
        mb.showinfo('提示','10ms超时模式')
        gmt(self.obj,512,10)
        gmt(self.obj,1024,10)
        gmt(self.obj,2048,10)
        mb.showinfo('提示','5ms超时模式')
        gmt(self.obj,512,5)
        gmt(self.obj,1024,5)
        gmt(self.obj,2048,5)
        mb.showinfo('提示','3ms超时模式')
        gmt(self.obj,512,3)
        gmt(self.obj,1024,3)
        gmt(self.obj,2048,3)
        mb.showinfo('提示','1ms超时模式')
        gmt(self.obj,512,1)
        gmt(self.obj,1024,1)
        gmt(self.obj,2048,1)
rt=tkinter.Tk()
rt.title('绎辰电脑跑分1.0(仅支持GPU)')
rt.geometry('600x600+150+100')
y=10
gpus=list_gpus()
gpulst=[]
print(gpus)
for i in gpus:
    gpgpu=GPUs(y,i)
    gpgpu.update()
    gpulst.append(gpgpu)
    y+=50
while True:
    try:
        rt.update()
    except tkinter.TclError:
        os._exit(0)
    except:
        break
