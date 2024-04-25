#!/usr/bin/python3
# coding=utf8
# Date:2022/05/30
import sys
import cv2
import time
import math
import rospy
import numpy as np
from threading import RLock, Timer, Thread

from std_srvs.srv import *
from std_msgs.msg import *
from sensor_msgs.msg import Image

from sensor.msg import Led
from chassis_control.msg import *
from visual_processing.msg import Result
from visual_processing.srv import SetParam
from intelligent_transport.srv import SetTarget
from hiwonder_servo_msgs.msg import MultiRawIdPosDur

from armpi_pro import PID
from armpi_pro import Misc
from armpi_pro import bus_servo_control
from kinematics import ik_transform

# 自主搬运

lock = RLock()
ik = ik_transform.ArmIK()

set_visual = 'line'   
detect_step = 'color' # 步骤：巡线或者检测色块
line_color = 'yellow' # 巡线颜色
stable = False        # 色块夹取判断变量
place_en = False      # 色块放置判断变量
position_en = False   # 色块夹取前定位判断变量
__isRunning = False   # 玩法控制开关变量
block_clamp = False   # 搬运色块标记变量
chassis_move = False  # 底盘移动标记变量

x_dis = 500
y_dis = 0.15
line_width = 0
line_center_x = 0
line_center_y = 0
color_centreX = 320
color_centreY = 410
color_center_x = 0
color_center_y = 0
detect_color = 'None'  
target_color = 'None'

img_h, img_w = 480, 640

line_x_pid = PID.PID(P=0.002, I=0.001, D=0)  # pid初始化
color_x_pid = PID.PID(P=0.06, I=0, D=0) 
color_y_pid = PID.PID(P=0.00003, I=0, D=0)

range_rgb = {
    'red': (0, 0, 255),
    'blue': (255, 0, 0),
    'green': (0, 255, 0),
    'black': (0, 0, 0),
    'yellow': (0, 255, 255),
    'white': (255, 255, 255),
}

def detect_yellow_lines_from_camera():
    # Initialize camera
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame from BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define range of yellow color in HSV
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])

        # Threshold the HSV image to get only yellow colors
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # Bitwise-AND mask and original image
        yellow_res = cv2.bitwise_and(frame, frame, mask=yellow_mask)

        # Convert yellow result to grayscale
        yellow_gray = cv2.cvtColor(yellow_res, cv2.COLOR_BGR2GRAY)

        # Apply edge detection using Canny
        edges = cv2.Canny(yellow_gray, 50, 150, apertureSize=3)

        # Perform Hough Line Transform
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

        if lines is not None:
            for rho, theta in lines[0]:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Display the resulting frame
        cv2.imshow('Frame', frame)

        # Wait for a short period of time (30 milliseconds)
        # to allow the display to refresh
        cv2.waitKey(30)

    # Release the camera and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

def detect_color_cubes_from_camera():
    # Initialize camera
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame from BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define range of green color in HSV
        lower_green = np.array([40, 40, 40])
        upper_green = np.array([80, 255, 255])

        # Define range of blue color in HSV
        lower_blue = np.array([100, 100, 100])
        upper_blue = np.array([140, 255, 255])

        # Define range of red color in HSV (split into two ranges due to hue wrapping)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])

        # Threshold the HSV image to get each color region
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)

        # Bitwise-AND mask and original image for each color
        green_res = cv2.bitwise_and(frame, frame, mask=green_mask)
        blue_res = cv2.bitwise_and(frame, frame, mask=blue_mask)
        red_res = cv2.bitwise_and(frame, frame, mask=red_mask)

        # Find contours for each color region
        green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw bounding boxes around the detected contours
        for contour in green_contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        for contour in blue_contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        for contour in red_contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Display the resulting frame
        cv2.imshow('Frame', frame)

        # Wait for a short period of time (30 milliseconds)
        # to allow the display to refresh
        cv2.waitKey(30)

    # Release the camera and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# 初始位置
def initMove(delay=True):
    with lock:
        bus_servo_control.set_servos(joints_pub, 1500, ((1, 75), (2, 500), (3, 80), (4, 825), (5, 625), (6, 500)))
    if delay:
        rospy.sleep(2)

# 关闭RGB灯
def off_rgb():
    global rgb_pub
    
    led = Led()
    led.index = 0
    led.rgb.r = 0
    led.rgb.g = 0
    led.rgb.b = 0
    rgb_pub.publish(led)
    led.index = 1
    rgb_pub.publish(led)


# 变量重置
def reset():
    global x_dis,y_dis
    global position_en,stable
    global set_visual,detect_step,place_en
    global block_clamp,chassis_move,target_color
    global line_width,line_center_x,line_center_y
    global detect_color,color_center_x,color_center_y
    
    with lock:
        line_x_pid.clear()
        color_x_pid.clear()
        color_y_pid.clear()
        off_rgb()
        set_visual = 'line'
        detect_step = 'color'
        stable = False
        place_en = False
        position_en = False
        block_clamp = False
        chassis_move = False
        x_dis = 500
        y_dis = 0.15
        line_width = 0
        line_center_x = 0
        line_center_y = 0
        color_center_x = 0
        color_center_y = 0
        detect_color = 'None'
        target_color = 'None'
        set_velocity.publish(0, 90, 0)
        

# 初始化调用
def init():
    rospy.loginfo("intelligent transport Init")
    initMove()
    reset()


n = 0
last_x = 0
last_y = 0

# Modify the run function to use detect_yellow_lines and detect_color_cubes
def run(msg):
    global lock
    global line_width, line_center_x, line_center_y
    global detect_color, color_center_x, color_center_y

    with lock:
        # Call detect_yellow_lines for line detection
        line_results = detect_yellow_lines(msg)
        if line_results is not None:
            line_width, line_center_x, line_center_y = line_results

        # Call detect_color_cubes for color detection
        color_results = detect_color_cubes(msg)
        if color_results is not None:
            detect_color, color_center_x, color_center_y = color_results
            
# 机器人移动函数
def move():
    global x_dis,y_dis
    global position_en,stable
    global set_visual,detect_step,place_en
    global block_clamp,chassis_move,target_color
    global line_width,line_center_x,line_center_y
    global detect_color,color_center_x,color_center_y
    
    num = 0
    transversae_num = 0
    move_time = time.time()
    place_delay = time.time()
    transversae_time = time.time()
    position = {'red':1, 'green':2, 'blue':3, 'None':-1} # 色块对应位置横线数
    
    while __isRunning:
        if detect_step == 'line': # 巡线阶段
            if set_visual == 'color': 
                set_visual = 'line'
                place_en = False
                visual_running('line', line_color) # 切换图像处理类型
                # 切换机械臂姿态
                bus_servo_control.set_servos(joints_pub, 1500, ((1, 500), (2, 500), (3, 80), (4, 825), (5, 625), (6, 500)))
                rospy.sleep(1.5)
                
            elif line_width > 0: #识别到线条
                # PID算法巡线
                if abs(line_center_x - img_w/2) < 30:
                    line_center_x = img_w/2
                line_x_pid.SetPoint = img_w/2      # 设定
                line_x_pid.update(line_center_x)   # 当前
                dx = round(line_x_pid.output, 2)   # 输出
                dx = 0.8 if dx > 0.8 else dx
                dx = -0.8 if dx < -0.8 else dx
                
                set_velocity.publish(100, 90, dx) # 控制底盘
                chassis_move = True
                
                if not place_en:
                    if line_width > 100 and block_clamp:  # 在夹取着色块时检测横线
                        if (time.time()-transversae_time) > 1:
                            transversae_num += 1
                            print(transversae_num)
                            transversae_time = time.time()
                    
                        if transversae_num == position[target_color]: # 判断当前横线数量是否等于目标颜色对应的数量
                            place_en = True  # 放置使能
                            if transversae_num == 1:
                                place_delay = time.time() + 1.1 # 设置延时停下来时间
                            elif transversae_num == 2:
                                place_delay = time.time() + 1.1
                            elif transversae_num == 3:
                                place_delay = time.time() + 1.2
                            
                elif place_en:
                    if time.time() >= place_delay: # 延时停下来，把色块放到横线旁边
                        rospy.sleep(0.1)
                        set_velocity.publish(0, 0, 0)
                        target = ik.setPitchRanges((-0.24, 0.00, -0.04), -180, -180, 0) #机械臂移动到色块放置位置
                        if target:
                            servo_data = target[1]
                            bus_servo_control.set_servos(joints_pub, 1200, ((6, servo_data['servo6']),)) 
                            rospy.sleep(1)
                            bus_servo_control.set_servos(joints_pub, 1500, ((3, servo_data['servo3']), (4, servo_data['servo4']), (5, servo_data['servo5'])))
                        rospy.sleep(1.8)

                        bus_servo_control.set_servos(joints_pub, 500, ((1, 150),))  # 张开机械爪
                        rospy.sleep(0.8)
                        
                        #机械臂复位
                        bus_servo_control.set_servos(joints_pub, 1500, ((1, 75), (2, 500), (3, 80), (4, 825), (5, 625)))
                        rospy.sleep(1.5)
                        bus_servo_control.set_servos(joints_pub, 1500, ((6, 500),))
                        rospy.sleep(1.5)
                        
                        move_time = time.time() + (11.5 - transversae_num) # 设置放置色块后要巡线的时间，让机器人回到初始位置
                            
                        # 变量重置
                        place_en = False
                        block_clamp = False
                        target_color = 'None'
                        set_rgb('black')
                        transversae_num = 0
                
                if not block_clamp and time.time() >= move_time: # 放置色块后机器人巡线回到初始位置
                    rospy.sleep(0.1)
                    set_velocity.publish(0, 0, 0)
                    detect_step = 'color'
                    
            else:
                if chassis_move:
                    chassis_move = False
                    rospy.sleep(0.1)
                    set_velocity.publish(0, 0, 0)
                else:
                    rospy.sleep(0.01)
            
            
        elif detect_step == 'color': # 色块检测阶段
            if set_visual == 'line':
                x_dis = 500
                y_dis = 0.15
                stable = False
                set_visual = 'color'
                visual_running('colors', 'rgb') # 切换图像处理类型
                # 切换机械臂姿态
                target = ik.setPitchRanges((0, 0.15, 0.03), -180, -180, 0)
                if target:
                    servo_data = target[1]
                    bus_servo_control.set_servos(joints_pub, 1500, ((1, 200), (2, 500), (3, servo_data['servo3']),
                                    (4, servo_data['servo4']),(5, servo_data['servo5']),(6, servo_data['servo6'])))
                    rospy.sleep(1.5)
                
            elif detect_color != 'None' and not block_clamp: # # 色块已经放稳，进行追踪夹取
                if position_en:
                    diff_x = abs(color_center_x - color_centreX)
                    diff_y = abs(color_center_y - color_centreY)
                    # X轴PID追踪
                    if diff_x < 10:
                        color_x_pid.SetPoint = color_center_x  # 设定
                    else:
                        color_x_pid.SetPoint = color_centreX
                    color_x_pid.update(color_center_x)   # 当前
                    dx = color_x_pid.output              # 输出
                    x_dis += int(dx)     
                    x_dis = 200 if x_dis < 200 else x_dis
                    x_dis = 800 if x_dis > 800 else x_dis
                    # Y轴PID追踪
                    if diff_y < 10:
                        color_y_pid.SetPoint = color_center_y  # 设定
                    else:
                        color_y_pid.SetPoint = color_centreY
                    color_y_pid.update(color_center_y)   # 当前
                    dy = color_y_pid.output              # 输出
                    y_dis += dy  
                    y_dis = 0.12 if y_dis < 0.12 else y_dis
                    y_dis = 0.28 if y_dis > 0.28 else y_dis
                    
                    # 机械臂追踪移动到色块上方    
                    target = ik.setPitchRanges((0, round(y_dis, 4), 0.03), -180, -180, 0)
                    if target:
                        servo_data = target[1]
                        bus_servo_control.set_servos(joints_pub, 20,((3, servo_data['servo3']),         
                             (4, servo_data['servo4']),(5, servo_data['servo5']), (6, x_dis)))
                    
                    if dx < 2 and dy < 0.003 and not stable: # 等待机械臂稳定停在色块上方
                        num += 1
                        if num == 10:
                            stable = True  # 设置可以夹取
                            num = 0
                    else:
                        num = 0
                    
                    if stable: #控制机械臂进行夹取
                        offset_y = Misc.map(target[2], -180, -150, -0.03, 0.03)
                        set_rgb(detect_color)       # 设置rgb灯颜色
                        target_color = detect_color # 暂存目标颜色
                        buzzer_pub.publish(0.1) #蜂鸣器响一下
                        
                        bus_servo_control.set_servos(joints_pub, 500, ((1, 120),)) #张开机械爪
                        rospy.sleep(0.5)
                        target = ik.setPitchRanges((0, round(y_dis + offset_y, 5), -0.07), -180, -180, 0) #机械臂向下伸
                        if target:
                            servo_data = target[1]
                            bus_servo_control.set_servos(joints_pub, 1000, ((3, servo_data['servo3']),
                                    (4, servo_data['servo4']),(5, servo_data['servo5']), (6, x_dis)))
                        rospy.sleep(1.5)
                        bus_servo_control.set_servos(joints_pub, 500, ((1, 500),)) #闭合机械爪
                        rospy.sleep(0.8)
                        
                        bus_servo_control.set_servos(joints_pub, 1500, ((1, 500), (2, 500), (3, 80), (4, 825), (5, 625), (6, 500))) #机械臂抬起来
                        rospy.sleep(1.5)
                        
                        stable = False
                        block_clamp = True
                        position_en = False
                        detect_step = 'line'
                    
                else:
                    rospy.sleep(0.01)
                            
        else:
            rospy.sleep(0.01)

# 设置RGB灯颜色
def set_rgb(color):
    global lock
    with lock:
        led = Led()
        led.index = 0
        led.rgb.r = range_rgb[color][2]
        led.rgb.g = range_rgb[color][1]
        led.rgb.b = range_rgb[color][0]
        rgb_pub.publish(led)
        rospy.sleep(0.05)
        led.index = 1
        rgb_pub.publish(led)
        rospy.sleep(0.1)


result_sub = None
heartbeat_timer = None
# enter服务回调函数
def enter_func(msg):
    global lock
    global result_sub
    
    rospy.loginfo("enter intelligent transport")
    init()
    with lock:
        if result_sub is None:
            rospy.ServiceProxy('/visual_processing/enter', Trigger)()
            result_sub = rospy.Subscriber('/visual_processing/result', Result, run)
            
    return [True, 'enter']

# exit服务回调函数
def exit_func(msg):
    global lock
    global result_sub
    global __isRunning
    global heartbeat_timer
    
    rospy.loginfo("exit intelligent transport")
    with lock:
        rospy.ServiceProxy('/visual_processing/exit', Trigger)()
        __isRunning = False
        reset()
        try:
            if result_sub is not None:
                result_sub.unregister()
                result_sub = None
            if heartbeat_timer is not None:
                heartbeat_timer.cancel()
                heartbeat_timer = None
                
        except BaseException as e:
            rospy.loginfo('%s', e)
        
    return [True, 'exit']

# Modify start_running to start the appropriate detection function
def start_running():
    global lock
    global __isRunning

    rospy.loginfo("start running intelligent transport")
    with lock:
        init()
        __isRunning = True
        rospy.sleep(0.1)
        # Start the appropriate detection function based on detect_step
        if detect_step == 'line':
            th = Thread(target=detect_yellow_lines)
        elif detect_step == 'color':
            th = Thread(target=detect_color_cubes)
        th.setDaemon(True)
        th.start()

# Modify stop_running to stop the appropriate detection function
def stop_running():
    global lock
    global __isRunning

    rospy.loginfo("stop running intelligent transport")
    with lock:
        reset()
        __isRunning = False
        initMove(delay=False)
        set_velocity.publish(0, 0, 0)
        rospy.ServiceProxy('/visual_processing/set_running', SetParam)()


# set_running服务回调函数
def set_running(msg):
    
    if msg.data:
        start_running()
    else:
        stop_running()
        
    return [True, 'set_running']

# heartbeat服务回调函数
def heartbeat_srv_cb(msg):
    global heartbeat_timer

    if isinstance(heartbeat_timer, Timer):
        heartbeat_timer.cancel()
    if msg.data:
        heartbeat_timer = Timer(5, rospy.ServiceProxy('/intelligent_transport/exit', Trigger))
        heartbeat_timer.start()
    rsp = SetBoolResponse()
    rsp.success = msg.data

    return rsp


if __name__ == '__main__':
    # 初始化节点
    rospy.init_node('intelligent_transport', log_level=rospy.DEBUG)
    # 视觉处理
    visual_running = rospy.ServiceProxy('/visual_processing/set_running', SetParam)
    # 舵机发布
    joints_pub = rospy.Publisher('/servo_controllers/port_id_1/multi_id_pos_dur', MultiRawIdPosDur, queue_size=1)
    # app通信服务
    enter_srv = rospy.Service('/intelligent_transport/enter', Trigger, enter_func)
    exit_srv = rospy.Service('/intelligent_transport/exit', Trigger, exit_func)
    running_srv = rospy.Service('/intelligent_transport/set_running', SetBool, set_running)
    heartbeat_srv = rospy.Service('/intelligent_transport/heartbeat', SetBool, heartbeat_srv_cb)
    # 麦轮底盘控制
    set_velocity = rospy.Publisher('/chassis_control/set_velocity', SetVelocity, queue_size=1)
    set_translation = rospy.Publisher('/chassis_control/set_translation', SetTranslation, queue_size=1)
    # 蜂鸣器
    buzzer_pub = rospy.Publisher('/sensor/buzzer', Float32, queue_size=1)
    # rgb 灯
    rgb_pub = rospy.Publisher('/sensor/rgb_led', Led, queue_size=1)
    rospy.sleep(0.5) # pub之后必须延时才能生效

    debug = False
    if debug:
        rospy.sleep(0.2)
        enter_func(1)
        start_running()
    
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down")

