import cv2
import numpy as np

def rescale_frame(frame, percent):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)


def init_video(url):
    video = cv2.VideoCapture(url)
    return video

def mask_1(frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
          #                      pos sabe el formato
        lower_limit = np.array([17,0,117])
        upper_limit = np.array([45,114,202])
        mask = cv2.inRange(hsv, lower_limit, upper_limit)
        res = cv2.bitwise_and(frame, frame, mask=mask)
        return [mask,res]

def morpho_trans(frame,k1,k2, cycles):
    kernel1 = np.ones((k1,k1),np.uint8)
    kernel2 = np.ones((k2,k2),np.uint8)
    for i in range(cycles-1):
        k2 +=1
        #k1 -=1
        erosion = cv2.erode(frame,kernel1)
        dilatacion =cv2.dilate(erosion,kernel2)
        frame = dilatacion
    
    return frame

def main():
    frames_counter = 1
    url   = "Computer Coding Challenge 1.mp4"
    video = init_video(url)

    while(True):
        
        frames_counter = frames_counter + 1
        
        # Capturar frame por frame del video
        ret, frame = video.read()
        if (ret==False):
            break

        #Reescalar la imagen
        frame = rescale_frame(frame,percent=70)

        #Frame con mascara aplicada
        [mask, bg_off] = mask_1(frame)

        #Video en escalas de grises
        gray = cv2.cvtColor(bg_off, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 130,255,cv2.THRESH_BINARY)

        #Transfomacion Morfologica
        morph = morpho_trans(binary,2,2,10)

        gauss_th = cv2.adaptiveThreshold(morph,255,cv2.cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)   
        contours, hierarchy = cv2.findContours(morph.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        #Comprobar que se trata de un archivo de video valido o que no este vacio
        if (frame is not None) and (binary is not None) and (morph is not None)and (mask is not None) :
            #reproducir video
            #===============================================
            # Video crudo
            cv2.drawContours(frame, contours, -1, (0,0,255), 3)
            cv2.imshow('frame',frame)
            # Video binario
            #cv2.imshow('binary image', binary)
            #Video con Mascara binario
            cv2.imshow('mask', mask)
            #Video con erosion
            #Video con Mascara a color
            #cv2.imshow('bg_off', bg_off)
            cv2.imshow('Filtros morfologicos', morph)
            cv2.imshow('Gauss Threshold', gauss_th)
            #key = cv2.waitKey(1)
            
            # Presiona q para salir
            if cv2.waitKey(22) & 0xFF == ord('q'):
                break
    
        else:
            print("Frame is None") 
            break

    print("Number of frames in the video: ", frames_counter)
    video.release()
    cv2.destroyAllWindows()
    print("Video stop")

if __name__ == "__main__":
    main()