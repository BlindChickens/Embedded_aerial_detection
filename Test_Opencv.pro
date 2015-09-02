#-------------------------------------------------
#
# Project created by QtCreator 2015-04-12T05:11:22
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = Test_Opencv
TEMPLATE = app

INCLUDEPATH += C:/opencv-mingw/install/include
LIBS += -LC:\\opencv-mingw\\install\\x64\\mingw\\bin \
    libopencv_core2410 \
    libopencv_highgui2410 \
    libopencv_imgproc2410 \
    libopencv_ml2410 \


SOURCES += main.cpp\
        mainwindow.cpp \
    thermalprocessing.cpp \
    feature_extraction.cpp \
    colourprocessing.cpp \
    training.cpp \
    videocapture.cpp

HEADERS  += mainwindow.h \
    main.h \
    thermalprocessing.h \
    feature_extraction.h \
    colourprocessing.h \
    training.h \
    videocapture.h
