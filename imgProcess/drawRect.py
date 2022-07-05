# -*- coding: utf-8 -*-
import os
import glob
import copy
import cv2
import re
import numpy as np

WIN_NAME = 'draw_rect'


class Rect(object):
    def __init__(self):
        self.tl = (0, 0)
        self.br = (0, 0)

    def regularize(self):
        """
        make sure tl = TopLeft point, br = BottomRight point
        """
        pt1 = (min(self.tl[0], self.br[0]), min(self.tl[1], self.br[1]))
        pt2 = (max(self.tl[0], self.br[0]), max(self.tl[1], self.br[1]))
        self.tl = pt1
        self.br = pt2


class DrawRects(object):
    def __init__(self, image, color, thickness=1):
        self.original_image = image
        self.image_for_show = image.copy()
        self.color = color
        self.thickness = thickness
        self.rects = Rect()
        self.current_rect = Rect()
        self.left_button_down = False

    @staticmethod
    def __clip(value, low, high):
        """
        clip value between low and high

        Parameters
        ----------
        value: a number
            value to be clipped
        low: a number
            low limit
        high: a number
            high limit

        Returns
        -------
        output: a number
            clipped value
        """
        output = max(value, low)
        output = min(output, high)
        return output

    def shrink_point(self, x, y):
        """
        shrink point (x, y) to inside image_for_show

        Parameters
        ----------
        x, y: int, int
            coordinate of a point

        Returns
        -------
        x_shrink, y_shrink: int, int
            shrinked coordinate
        """
        height, width = self.image_for_show.shape[0:2]
        x_shrink = self.__clip(x, 0, width)
        y_shrink = self.__clip(y, 0, height)
        return (x_shrink, y_shrink)

    def append(self):
        """
        add a rect to rects list
        """
        self.rects = copy.deepcopy(self.current_rect)

    def pop(self):
        """
        pop a rect from rects list
        """
        self.rects = Rect()
        return self.rects

    def reset_image(self):
        """
        reset image_for_show using original image
        """
        self.image_for_show = self.original_image.copy()

    def draw(self):
        """
        draw rects on image_for_show
        """
        cv2.rectangle(self.image_for_show, self.rects.tl, self.rects.br,
                          color=self.color, thickness=self.thickness)

    def draw_current_rect(self):
        """
        draw current rect on image_for_show
        """
        cv2.rectangle(self.image_for_show,
                      self.current_rect.tl, self.current_rect.br,
                      color=self.color, thickness=self.thickness)


def onmouse_draw_rect(event, x, y, flags, draw_rects):
    if event == cv2.EVENT_LBUTTONDOWN:
        # pick first point of rect
        print('pt1: x = %d, y = %d' % (x, y))
        draw_rects.left_button_down = True
        draw_rects.current_rect.tl = (x, y)
    if draw_rects.left_button_down and event == cv2.EVENT_MOUSEMOVE:
        # pick second point of rect and draw current rect
        draw_rects.current_rect.br = draw_rects.shrink_point(x, y)
        draw_rects.reset_image()
        # draw_rects.draw()
        draw_rects.draw_current_rect()
    if event == cv2.EVENT_LBUTTONUP:
        # finish drawing current rect and append it to rects list
        draw_rects.left_button_down = False
        draw_rects.current_rect.br = draw_rects.shrink_point(x, y)
        print('pt2: x = %d, y = %d' % (draw_rects.current_rect.br[0],
                                       draw_rects.current_rect.br[1]))
        draw_rects.current_rect.regularize()
        draw_rects.append()
    if (not draw_rects.left_button_down) and event == cv2.EVENT_RBUTTONDOWN:
        # pop the last rect in rect s list
        draw_rects.pop()
        draw_rects.reset_image()
        draw_rects.draw()


def hand_crop(imagePath):
    images =[os.path.join(imagePath + "testFigures",path) for path in sorted(os.listdir(imagePath + "testFigures"),key = lambda i:int(re.match(r'(\d+)',i).group())) ] 
    idx = 0
    draw_rects = DrawRects(cv2.imread(images[idx]), (0, 255, 0), 1)
    cv2.namedWindow(WIN_NAME, 0)
    cv2.setMouseCallback(WIN_NAME, onmouse_draw_rect, draw_rects)
    while True:
        cv2.imshow(WIN_NAME, draw_rects.image_for_show)
        key = cv2.waitKey(30)
        # print(key)
        if key == 27:  # ESC
            break
        if key == 97:
            if idx > 0:
                idx = idx - 1
            draw_rects.original_image = cv2.imread(images[idx])
            draw_rects.reset_image()
            print(images[idx])
        if key == 100:
            if idx < len(images)-1:
                idx = idx + 1
            draw_rects.original_image = cv2.imread(images[idx])
            draw_rects.reset_image()
            print(images[idx])
    cv2.destroyAllWindows()
    mask = np.zeros(draw_rects.original_image.shape[:2], dtype="uint8")
    cv2.rectangle(mask, draw_rects.rects.tl, draw_rects.rects.br,
                      color=(255,255,255), thickness=-1)
    return mask


if __name__ == '__main__':
    image = np.zeros((480, 640, 3), np.uint8)
    draw_rects = DrawRects(image, (0, 255, 0), 1)
    cv2.namedWindow(WIN_NAME, 0)
    cv2.setMouseCallback(WIN_NAME, onmouse_draw_rect, draw_rects)
    while True:
        cv2.imshow(WIN_NAME, draw_rects.image_for_show)
        key = cv2.waitKey(30)
        if key == 27:  # ESC
            break
    cv2.destroyAllWindows()