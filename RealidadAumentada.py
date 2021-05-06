import cv2 
import numpy as np 

cap = cv2.VideoCapture(0)

imgTarget= cv2.imread('imagenPrueba.jpeg')
imgTarget= cv2.resize(imgTarget,(500,500))
#video que vamos a mostrar
myVid= cv2.VideoCapture('video.mp4')

detection=False
frameCounter=0


success, imgVideo = myVid.read()
hT, wT, cT = imgTarget.shape
imgVideo = cv2.resize(imgVideo, (wT, hT))

orb = cv2.ORB_create(nfeatures=1000)
kp1, des1 = orb.detectAndCompute(imgTarget, None)
#tomamos los puntos importantes de la imagen
#imgTarget=cv2.drawKeypoints(imgTarget,kp1,None)



def stackImages(imgArray,scale,lables=[]):
    sizeW= imgArray[0][0].shape[1]
    sizeH = imgArray[0][0].shape[0]
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (sizeW,sizeH), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((sizeH, sizeW, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (sizeW, sizeH), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        hor_con= np.concatenate(imgArray)
        ver = hor
    if len(lables) != 0:
        eachImgWidth= int(ver.shape[1] / cols)
        eachImgHeight = int(ver.shape[0] / rows)
        print(eachImgHeight)
        for d in range(0, rows):
            for c in range (0,cols):
                cv2.rectangle(ver,(c*eachImgWidth,eachImgHeight*d),(c*eachImgWidth+len(lables[d][c])*13+27,30+eachImgHeight*d),(255,255,255),cv2.FILLED)
                cv2.putText(ver,lables[d][c],(eachImgWidth*c+10,eachImgHeight*d+20),cv2.FONT_HERSHEY_COMPLEX,0.7,(255,0,255),2)
    return ver






while True:


    success, imgWebcam = cap.read()
    #vamos a comparar los puntos de importantes de la imagen , a ver si aparecen en el video
    imgAug = imgWebcam.copy()
    kp2, des2 = orb.detectAndCompute(imgWebcam, None)
    #tomamos los puntos importantes del video
    #imgWebcam= cv2.drawKeypoints(imgWebcam,kp2,None)

    if detection == False:
        #si no hay deteccion el video lo vamos a resetear
        myVid.set(cv2.CAP_PROP_POS_FRAMES,0)
        frameCounter=0

    else:
        #si lo detecto
        if frameCounter == myVid.get(cv2.CAP_PROP_FRAME_COUNT):
            myVid.set(cv2.CAP_PROP_POS_FRAMES,0)
            frameCounter=0

        success,imgVideo= myVid.read()
        imgVideo=cv2.resize(imgVideo,(wT,hT))

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        #calculo la distancia
        if m.distance < 0.75 * n.distance:
            good.append(m)
    print(len(good))

    #ya detectamos la cantidad de matches
    imgFeatures = cv2.drawMatches(imgTarget, kp1, imgWebcam, kp2, good, None, flags=2)
    
    #si los puntos buenos superan un umbral
    if len(good)>18:
        detection=True
        #aca estamos haciendo un loop y buscando los buenos matches
        srcPts= np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dstPts= np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

        matrix, mask= cv2.findHomography(srcPts,dstPts,cv2.RANSAC,5)
        print(matrix)

        #vamos a buscar el box que vamos a plotear
        pts=np.float32([[0,0],[0,hT],[wT,hT],[wT,0]]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,matrix)
        img2= cv2.polylines(imgWebcam,[np.int32(dst)],True,(255,0,255),3)

        #
        imgWarp = cv2.warpPerspective(imgVideo, matrix, (img2.shape[1], img2.shape[0]))

        #vamos a crear una mascara
        maskNew = np.zeros((imgAug.shape[0],imgAug.shape[1]),np.uint8)
        cv2.fillPoly(maskNew,[np.int32(dst)],(255,255,255))

        maskInv = cv2.bitwise_not(maskNew) 
        imgAug = cv2.bitwise_and(imgAug, imgAug, mask=maskInv)
        #agrego el video en el contorno negro
        imgAug = cv2.bitwise_or(imgWarp, imgAug) 

        #llamo la funcion strackImages() me va a devolver todo en un solo visor
        #imgStacked= stackImages(([imgWebcam,imgVideo,imgTarget],[imgFeatures,imgWarp,imgAug]),0.1)

    cv2.imshow('imgAug',imgAug)
    #cv2.imshow('imgWarp',imgWarp)
    #cv2.imshow('img2',img2)
    #cv2.imshow('imgFeatures',imgFeatures)
    #cv2.imshow('ImgTarget',imgTarget)
    #cv2.imshow('ImgVideo',imgVideo)
    #cv2.imshow('ImgWebCam',imgWebcam)'''
    #cv2.imshow('imgStacked',imgStacked)
    cv2.waitKey(1)
    frameCounter+=1