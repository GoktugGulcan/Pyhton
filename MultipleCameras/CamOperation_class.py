from collections import deque
import threading
import time
import tkinter as tk
import cv2
import sys
import logging
import numpy as np
from PIL import Image, ImageTk
from queue import Queue
from ctypes import cdll, CFUNCTYPE
import gc
sys.path.append("/home/alfa/Downloads/Python5/Python/MvImport")
from MvCameraControl_class import *
from ultralytics import YOLO

# Set up logging
logging.basicConfig(level=logging.INFO)

class CameraOperation:
    def __init__(self, cam=None, st_device_list=None, n_connect_num=0):
        self.lock = threading.Lock()
        self.N = 5
        self.init_camera_attributes(cam, st_device_list, n_connect_num)
        self.init_yolo_model()
        self.confidence_threshold = 0.5

    def init_camera_attributes(self, cam, st_device_list, n_connect_num):
        """Initialize camera attributes."""
        self.cam = cam
        self.st_device_list = st_device_list
        self.n_connect_num = n_connect_num
        self.prev_detections = deque(maxlen=self.N)  # holds the last N frames of detections
        self.processed_queue = Queue()
        self.processed_image = None
        self.b_open_device = False
        self.b_start_grabbing = False
        self.b_thread_closed = False
        self.b_save_bmp = False
        self.b_save_jpg = False
        self.buf_save_image = None
        self.h_thread_handle = None
        self.n_save_image_size = 0
        self.st_frame_info = None
        self.b_exit = False
        self.image_queue = Queue()

    def init_yolo_model(self):
        """Initialize the YOLO model."""
        try:
            self.yolo_model = YOLO('/home/alfa/Downloads/Python5/Python/MultipleCameras/last.pt')
        except Exception as e:
            logging.error(f"Error initializing YOLO model: {e}")
            self.yolo_model = None

    @staticmethod
    def convert_yolo_to_opencv_bbox(bbox, img_width, img_height, height_reduction_factor=0.7, width_reduction_factor=0.7):
        width_factor = img_width / 640.0
        height_factor = img_height / 480.0
        
        x_center = bbox[0] * width_factor
        y_center = bbox[1] * height_factor
        width = bbox[2] * width_factor
        height = bbox[3] * height_factor
        
        x1 = int(x_center - (width / 2))
        y1 = int(y_center - (height / 2))
        x2 = int(x_center + (width / 2))
        y2 = int(y_center + (height / 2))
        
        # Adjust the height based on the height_reduction_factor
        height_adjustment = height * height_reduction_factor
        y1 = y1 + int(height_adjustment / 2)
        y2 = y2 - int(height_adjustment / 2)

        # Adjust the width based on the width_reduction_factor
        width_adjustment = width * width_reduction_factor
        x1 = x1 + int(width_adjustment / 2)
        x2 = x2 - int(width_adjustment / 2)
        
        # Adding an offset to shift the bounding box
        x_offset = int(0.05 * img_width)
        y_offset = int(0.05 * img_height)
        x1 += x_offset
        y1 += y_offset
        x2 += x_offset
        y2 += y_offset
        
        # Clipping the bounding box coordinates to ensure they stay within the image dimensions
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_width, x2)
        y2 = min(img_height, y2)

        return (x1, y1, x2, y2)


    def apply_nms(self, detections, iou_thresh=0.6):
        """
        Apply Non-Maximum Suppression on detections.
        
        :param detections: List of detections.
        :param iou_thresh: IoU threshold for NMS.
        :return: List of detections after NMS.
        """
        try:
            boxes = np.array([list(d[:4]) for d in detections])
            scores = np.array([d[4] for d in detections])
            indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), self.confidence_threshold, iou_thresh)
            return [detections[i[0] if isinstance(i, (list, tuple, np.ndarray)) else i] for i in indices]
        except Exception as e:
            logging.error(f"Error in improved NMS application: {str(e)}")
            return detections
        


    def annotate_image(self, img_np, debug=True):
        if img_np is None:
            logging.error("Image is None.")
            return img_np
        if self.yolo_model is None:
            logging.error("YOLO model not loaded.")
            return img_np

        resized_img_np = cv2.resize(img_np, (640, 480))
        results = self.yolo_model(resized_img_np)
        detections = results[0].boxes.data
        valid_detections = [x for x in detections if x[4] >= self.confidence_threshold]
        if len(valid_detections) == 0:
            print("No detections found.")
            return img_np

        valid_detections = self.apply_nms(valid_detections)
        for detection in valid_detections:
            bbox = self.convert_yolo_to_opencv_bbox(detection[:4], resized_img_np.shape[1], resized_img_np.shape[0])
            if bbox:
                class_id = int(detection[-1])
                class_name = self.yolo_model.names[class_id]

                # Setting color based on class name
                if class_name == "Lehimli":
                    color = (0, 255, 0)  # Green
                elif class_name == "Lehimsiz":
                    color = (0, 0, 255)  # Red
                else:
                    color = (0, 255, 0)  # Default to green if not any of the above

                cv2.rectangle(resized_img_np, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                self.draw_class_name(resized_img_np, class_name, bbox, color)
                        
                if debug:
                    print(f"Raw detection from YOLO: {detection[:4]}")
                    print(f"Converted bounding box: {bbox}")

        annotated_original_dim_img = cv2.resize(resized_img_np, (img_np.shape[1], img_np.shape[0]))
        return annotated_original_dim_img
       

    def draw_class_name(self, annotated_img, class_name, bbox, color):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1  # Adjusted font scale
        font_thickness = 2
        (text_width, text_height), _ = cv2.getTextSize(class_name, font, font_scale, font_thickness)
        
        if None in bbox:
            print("Error: Some values in bbox are None!")
            return

        # Adjusting the bounding box to be of the same size as the name
        text_box_top_left = (bbox[0] + 5, bbox[1] - text_height - 10)
        text_box_bottom_right = (bbox[0] + text_width + 10, bbox[1] - 10)
        
        # Draw the bounding box
        cv2.rectangle(annotated_img, text_box_top_left, text_box_bottom_right, color, 2)
        
        # Draw the colored background for text inside bounding box
        cv2.rectangle(annotated_img, text_box_top_left, text_box_bottom_right, color, cv2.FILLED)
        # Draw the class name in black over the colored background
        cv2.putText(annotated_img, class_name, (bbox[0] + 10, bbox[1] - 10), font, font_scale, (0, 0, 0), font_thickness)


    def resize_image(self, image, target_width, target_height):
        """
        Resize the given image to the specified width and height.
        """
        try:
            return cv2.resize(image, (target_width, target_height))
        except Exception as e:
            logging.error(f"Error resizing image: {e}")
            return image

   
    def process_frame(self, frame):
        logging.info("Beginning to process frame...")
        

        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        annotated_image = self.annotate_image(frame_rgb)
        
        # Check if the annotated image is correctly added to the queue
        if annotated_image is not None:
            self.processed_queue.put((frame_rgb, annotated_image))
            logging.info("Processed image added to queue.")
        else:
            logging.error("Annotated image is None. Not adding to queue.")

    def process_images(self):
        while not self.b_exit:
            try:
                ret, frame = self.cam.read()
                if not ret:
                    logging.error("Failed to grab frame")
                    continue

                self.process_frame(frame)  # Process and annotate the frame
                time.sleep(0.05)
            except Exception as e:
                logging.error(f"Error in process_images: {e}")



    @staticmethod
    def To_hex_str(num):
        if num is None:
            return "None"
        return hex(num & 0xFFFFFFFF)

    # 打开相机
    def Open_device(self):
        if self.b_open_device is False:
            # ch:选择设备并创建句柄 | en:Select device and create handle
            nConnectionNum = int(self.n_connect_num)
            stDeviceList = cast(self.st_device_list.pDeviceInfo[int(nConnectionNum)],
                                POINTER(MV_CC_DEVICE_INFO)).contents
            self.cam = MvCamera()
            ret = self.cam.MV_CC_CreateHandle(stDeviceList)
            if ret != 0:
                self.cam.MV_CC_DestroyHandle()
                return ret

            ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
            if ret != 0:
                self.b_open_device = False
                self.b_thread_closed = False
                return ret
            self.b_open_device = True
            self.b_thread_closed = False

            # ch:探测网络最佳包大小(只对GigE相机有效) | en:Detection network optimal package size(It only works for the GigE camera)
            if stDeviceList.nTLayerType == MV_GIGE_DEVICE:
                nPacketSize = self.cam.MV_CC_GetOptimalPacketSize()
                if int(nPacketSize) > 0:
                    ret = self.cam.MV_CC_SetIntValue("GevSCPSPacketSize", nPacketSize)
                    if ret != 0:
                        print("warning: set packet size fail! ret[0x%x]" % ret)
                else:
                    print("warning: packet size is invalid[%d]" % nPacketSize)

            stBool = c_bool(False)
            ret = self.cam.MV_CC_GetBoolValue("AcquisitionFrameRateEnable", stBool)
            if ret != 0:
                print("warning: get acquisition frame rate enable fail! ret[0x%x]" % ret)

            # ch:设置触发模式为off | en:Set trigger mode as off
            ret = self.cam.MV_CC_SetEnumValueByString("TriggerMode", "Off")
            if ret != 0:
                print("warning: set trigger mode off fail! ret[0x%x]" % ret)
            return 0

    # 开始取图
    def Start_grabbing(self, index, root, panel, lock):
        if False == self.b_start_grabbing and True == self.b_open_device:
            self.b_exit = False
            ret = self.cam.MV_CC_StartGrabbing()
            if ret != 0:
                self.b_start_grabbing = False
                return ret
            self.b_start_grabbing = True
            try:
                self.h_thread_handle = threading.Thread(target=CameraOperation.Work_thread,
                                                        args=(self, index, root, panel, lock))
                self.h_thread_handle.start()
                self.b_thread_closed = True
            except:
                tk.messagebox.showerror('show error', 'error: unable to start thread')
                self.b_start_grabbing = False
            return ret

    # 停止取图
    def Stop_grabbing(self):
        if True == self.b_start_grabbing and self.b_open_device == True:
            # 退出线程
            if self.b_thread_closed:
                self.b_exit = True
                # Stop_thread(self.h_thread_handle)
                self.b_thread_closed = False
            ret = self.cam.MV_CC_StopGrabbing()
            if ret != 0:
                self.b_start_grabbing = True
                return ret
            self.b_start_grabbing = False
            return 0

    # 关闭相机
    def Close_device(self):
        if self.b_open_device:
            # 退出线程
            if self.b_thread_closed:
                self.b_exit = False
                self.b_thread_closed = False
            ret = self.cam.MV_CC_StopGrabbing()
            ret = self.cam.MV_CC_CloseDevice()
            return ret

        # ch:销毁句柄 | Destroy handle
        self.cam.MV_CC_DestroyHandle()
        self.b_open_device = False
        self.b_start_grabbing = False

    # 设置触发模式
    def Set_trigger_mode(self, strMode):
        if True == self.b_open_device:
            if "continuous" == strMode:
                ret = self.cam.MV_CC_SetEnumValueByString("TriggerMode", "Off")
                if ret != 0:
                    return ret
                else:
                    return 0
            if "triggermode" == strMode:
                ret = self.cam.MV_CC_SetEnumValueByString("TriggerMode", "On")
                if ret != 0:
                    return ret
                ret = self.cam.MV_CC_SetEnumValueByString("TriggerSource", "Software")
                if ret != 0:
                    return ret
                return ret

    # 软触发一次
    def Trigger_once(self, nCommand):
        if True == self.b_open_device:
            if 1 == nCommand:
                ret = self.cam.MV_CC_SetCommandValue("TriggerSoftware")
                return ret

    # 获取参数
    def Get_parameter(self):
        if True == self.b_open_device:
            stFloatParam_FrameRate = MVCC_FLOATVALUE()
            memset(byref(stFloatParam_FrameRate), 0, sizeof(MVCC_FLOATVALUE))
            stFloatParam_exposureTime = MVCC_FLOATVALUE()
            memset(byref(stFloatParam_exposureTime), 0, sizeof(MVCC_FLOATVALUE))
            stFloatParam_gain = MVCC_FLOATVALUE()
            memset(byref(stFloatParam_gain), 0, sizeof(MVCC_FLOATVALUE))
            ret = self.cam.MV_CC_GetFloatValue("AcquisitionFrameRate", stFloatParam_FrameRate)
            self.frame_rate = stFloatParam_FrameRate.fCurValue
            ret = self.cam.MV_CC_GetFloatValue("ExposureTime", stFloatParam_exposureTime)
            self.exposure_time = stFloatParam_exposureTime.fCurValue
            ret = self.cam.MV_CC_GetFloatValue("Gain", stFloatParam_gain)
            self.gain = stFloatParam_gain.fCurValue
            return ret

    # 设置参数
    def Set_parameter(self, frameRate, exposureTime, gain):
        if '' == frameRate or '' == exposureTime or '' == gain:
            return -1
        if True == self.b_open_device:
            ret = self.cam.MV_CC_SetFloatValue("ExposureTime", float(exposureTime))
            ret = self.cam.MV_CC_SetFloatValue("Gain", float(gain))
            ret = self.cam.MV_CC_SetFloatValue("AcquisitionFrameRate", float(frameRate))
            return ret

    def Work_thread(self, index, root, panel, lock):
        logging.info(f"Camera[{index}]: Work thread started.")
        stFrameInfo = MV_FRAME_OUT_INFO_EX()
        memset(byref(stFrameInfo), 0, sizeof(stFrameInfo))
        buf_cache = None

        while not self.b_exit:
            start_time = time.time()  # Start the timer

            try:
                logging.info(f"Camera[{index}]: Attempting to get image buffer.")
                ret = self.cam.MV_CC_GetOneFrameTimeout(self, stFrameInfo, 5000)

                if ret != 0:
                    logging.error(f"Error in Work_thread for Camera[{index}]: Failed to get image buffer. Error code: {ret}")
                    continue

                logging.info(f"Camera[{index}]: Image buffer obtained.")
                logging.info(f"Camera[{index}]: Frame width: {stFrameInfo.nWidth}")
                logging.info(f"Camera[{index}]: Frame height: {stFrameInfo.nHeight}")
                logging.info(f"Camera[{index}]: Pixel type: {stFrameInfo.enPixelType}")

                if not stFrameInfo.pBufAddr:
                    logging.error(f"Camera[{index}]: pBufAddr is NULL!")
                    continue

                logging.info(f"Camera[{index}]: pBufAddr is valid.")

                if buf_cache is None or len(buf_cache) != stFrameInfo.stFrameInfo.nFrameLen:
                    buf_cache = (c_ubyte * stFrameInfo.stFrameInfo.nFrameLen)()
                    logging.info(f"Camera[{index}]: Buffer cache re-initialized.")

                ctypes.memmove(byref(buf_cache), stFrameInfo.pBufAddr, stFrameInfo.stFrameInfo.nFrameLen)
                logging.info(f"Camera[{index}]: Memory moved to buffer cache.")

                # Convert the frame to RGB format
                numArray_rgb = self.Mono_numpy(buf_cache, stFrameInfo.stFrameInfo.nWidth, stFrameInfo.stFrameInfo.nHeight) if stFrameInfo.stFrameInfo.enPixelType != PixelType_Gvsp_RGB8_Packed else self.Color_numpy(buf_cache, stFrameInfo.stFrameInfo.nWidth, stFrameInfo.stFrameInfo.nHeight)

                # Process the frame for detections
                numArray_bgr = cv2.cvtColor(numArray_rgb, cv2.COLOR_RGB2BGR)
                self.process_frame(numArray_bgr)

                # Check if there's a processed image ready for display
                if not self.processed_queue.empty():
                    _, processed_img = self.processed_queue.get()
                current_image = Image.fromarray(processed_img).resize((500, 500), Image.LANCZOS)
                with lock:
                    def update_gui():
                        imgtk = ImageTk.PhotoImage(image=current_image, master=root)
                        panel.imgtk = imgtk
                        panel.config(image=imgtk)
                        logging.info(f"Camera[{index}]: GUI updated with processed image.")

                    root.after(0, update_gui)

                stConvertParam = MV_CC_PIXEL_CONVERT_PARAM()
                memset(byref(stConvertParam), 0, sizeof(stConvertParam))
                stConvertParam.nWidth = stFrameInfo.stFrameInfo.nWidth
                stConvertParam.nHeight = stFrameInfo.stFrameInfo.nHeight
                stConvertParam.pSrcData = cast(buf_cache, POINTER(c_ubyte))
                stConvertParam.nSrcDataLen = stFrameInfo.stFrameInfo.nFrameLen
                stConvertParam.enSrcPixelType = stFrameInfo.stFrameInfo.enPixelType

                # Convert to RGB for display
                if PixelType_Gvsp_RGB8_Packed == stFrameInfo.stFrameInfo.enPixelType:
                    numArray = self.Color_numpy(buf_cache, stFrameInfo.stFrameInfo.nWidth, stFrameInfo.stFrameInfo.nHeight)
                else:
                    nConvertSize = stFrameInfo.stFrameInfo.nWidth * stFrameInfo.stFrameInfo.nHeight * 3 + 2048
                    stConvertParam.enDstPixelType = PixelType_Gvsp_RGB8_Packed
                    stConvertParam.pDstBuffer = (c_ubyte * nConvertSize)()
                    stConvertParam.nBufferSize = nConvertSize  # ch:存储节点的大小 | en:Buffer node size
                    ret = self.cam.MV_CC_ConvertPixelType(stConvertParam)
                    if ret != 0:
                        logging.error(f"Error in Work_thread: Failed to convert pixel type. Error code: {ret}")
                        continue
                    img_buff = (c_ubyte * nConvertSize)()
                    ctypes.memmove(byref(img_buff), stConvertParam.pDstBuffer, nConvertSize)
                    numArray = self.Color_numpy(img_buff, stFrameInfo.stFrameInfo.nWidth, stFrameInfo.stFrameInfo.nHeight)

                
                # nRet = self.cam.MV_CC_FreeImageForBGR(stFrameInfo)

                gc.collect()
                del buf_cache

            except Exception as e:
                logging.error(f"Exception in Work_thread for Camera[{index}]: {e}")



    # 存jpg图像
    def Save_jpg(self, buf_cache):
        if (None == buf_cache):
            return
        self.buf_save_image = None
        file_path = str(self.st_frame_info.nFrameNum) + ".jpg"
        self.n_save_image_size = self.st_frame_info.nWidth * self.st_frame_info.nHeight * 3 + 2048
        if self.buf_save_image is None:
            self.buf_save_image = (c_ubyte * self.n_save_image_size)()

        stParam = MV_SAVE_IMAGE_PARAM_EX()
        stParam.enImageType = MV_Image_Jpeg  # ch:需要保存的图像类型 | en:Image format to save
        stParam.enPixelType = self.st_frame_info.enPixelType  # ch:相机对应的像素格式 | en:Camera pixel type
        stParam.nWidth = self.st_frame_info.nWidth  # ch:相机对应的宽 | en:Width
        stParam.nHeight = self.st_frame_info.nHeight  # ch:相机对应的高 | en:Height
        stParam.nDataLen = self.st_frame_info.nFrameLen
        stParam.pData = cast(buf_cache, POINTER(c_ubyte))
        stParam.pImageBuffer = cast(byref(self.buf_save_image), POINTER(c_ubyte))
        stParam.nBufferSize = self.n_save_image_size  # ch:存储节点的大小 | en:Buffer node size
        stParam.nJpgQuality = 80  # ch:jpg编码，仅在保存Jpg图像时有效。保存BMP时SDK内忽略该参数
        return_code = self.cam.MV_CC_SaveImageEx2(stParam)

        if return_code != 0:
            tk.messagebox.showerror('show error', 'save jpg fail! ret = ' + self.To_hex_str(return_code))
            self.b_save_jpg = False
            return
        file_open = open(file_path.encode('ascii'), 'wb+')
        img_buff = (c_ubyte * stParam.nImageLen)()
        try:
            cdll.msvcrt.memcpy(byref(img_buff), stParam.pImageBuffer, stParam.nImageLen)
            file_open.write(img_buff)
            self.b_save_jpg = False
            tk.messagebox.showinfo('show info', 'save jpg success!')
        except Exception as e:
            self.b_save_jpg = False
            raise Exception("get one frame failed: %s" % str(e))
        if img_buff is not None:
            del img_buff
        if self.buf_save_image is not None:
            del self.buf_save_image

    # 存BMP图像
    def Save_Bmp(self, buf_cache):
        if (0 == buf_cache):
            return
        self.buf_save_image = None
        file_path = str(self.st_frame_info.nFrameNum) + ".bmp"
        self.n_save_image_size = self.st_frame_info.nWidth * self.st_frame_info.nHeight * 3 + 2048
        if self.buf_save_image is None:
            self.buf_save_image = (c_ubyte * self.n_save_image_size)()

        stParam = MV_SAVE_IMAGE_PARAM_EX()
        stParam.enImageType = MV_Image_Bmp  # ch:需要保存的图像类型 | en:Image format to save
        stParam.enPixelType = self.st_frame_info.enPixelType  # ch:相机对应的像素格式 | en:Camera pixel type
        stParam.nWidth = self.st_frame_info.nWidth  # ch:相机对应的宽 | en:Width
        stParam.nHeight = self.st_frame_info.nHeight  # ch:相机对应的高 | en:Height
        stParam.nDataLen = self.st_frame_info.nFrameLen
        stParam.pData = cast(buf_cache, POINTER(c_ubyte))
        stParam.pImageBuffer = cast(byref(self.buf_save_image), POINTER(c_ubyte))
        stParam.nBufferSize = self.n_save_image_size  # ch:存储节点的大小 | en:Buffer node size
        return_code = self.cam.MV_CC_SaveImageEx2(stParam)
        if return_code != 0:
            tk.messagebox.showerror('show error', 'save bmp fail! ret = ' + self.To_hex_str(return_code))
            self.b_save_bmp = False
            return
        file_open = open(file_path.encode('ascii'), 'wb+')
        img_buff = (c_ubyte * stParam.nImageLen)()
        try:
            cdll.msvcrt.memcpy(byref(img_buff), stParam.pImageBuffer, stParam.nImageLen)
            file_open.write(img_buff)
            self.b_save_bmp = False
            tk.messagebox.showinfo('show info', 'save bmp success!')
        except Exception as e:
            self.b_save_bmp = False
            raise Exception("get one frame failed: %s" % str(e))
        if img_buff is not None:
            del img_buff
        if self.buf_save_image is not None:
            del self.buf_save_image

   # Mono图像转为python数组
    def Mono_numpy(self, data, nWidth, nHeight):
        data_ = np.frombuffer(data, count=int(nWidth * nHeight), dtype=np.uint8, offset=0)
        data_mono_arr = data_.reshape(nHeight, nWidth)
        numArray = np.zeros([nHeight, nWidth, 1], "uint8")
        numArray[:, :, 0] = data_mono_arr
        return numArray

    # 彩色图像转为python数组
    def Color_numpy(self, data, nWidth, nHeight):
        data_ = np.frombuffer(data, count=int(nWidth * nHeight * 3), dtype=np.uint8, offset=0)
        data_r = data_[0:nWidth * nHeight * 3:3]
        data_g = data_[1:nWidth * nHeight * 3:3]
        data_b = data_[2:nWidth * nHeight * 3:3]

        data_r_arr = data_r.reshape(nHeight, nWidth)
        data_g_arr = data_g.reshape(nHeight, nWidth)
        data_b_arr = data_b.reshape(nHeight, nWidth)
        numArray = np.zeros([nHeight, nWidth, 3], "uint8")

        numArray[:, :, 0] = data_r_arr
        numArray[:, :, 1] = data_g_arr
        numArray[:, :, 2] = data_b_arr
        return numArray