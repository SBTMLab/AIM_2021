from bs4 import BeautifulSoup as bs
import requests
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import cv2

from classification_model import classification_model

background_color = (211, 200, 86)
font_color = (255, 255, 255)

model = classification_model()
light_magnitude = None
dust_before = None
dust_after = None

reference_pos = np.random.randint(low=0, high=10, size=(20, 2))
reference_neg = np.random.randint(low=20, high=30, size=(30, 2))

rivo_weight, dust_weight = 1, 2

while True:
    announce = np.full((700, 1000, 3), background_color, np.uint8)

    text = "Press 'u' to check pathogen in your mask"
    cv2.putText(announce, text, (50, 50), cv2.FONT_HERSHEY_TRIPLEX, 1, font_color, 2, cv2.LINE_AA)

    text = "Press 'd' to check dust in your mask"
    cv2.putText(announce, text, (50, 200), cv2.FONT_HERSHEY_TRIPLEX, 1, font_color, 2, cv2.LINE_AA)

    text = "Press 'k' to run weighted KNN algorithm"
    cv2.putText(announce, text, (50, 350), cv2.FONT_HERSHEY_TRIPLEX, 1, font_color, 2, cv2.LINE_AA)

    text = "Press 'w' to check today's dust info"
    cv2.putText(announce, text, (50, 500), cv2.FONT_HERSHEY_TRIPLEX, 1, font_color, 2, cv2.LINE_AA)

    text = "Press 'q' to exit"
    cv2.putText(announce, text, (50, 650), cv2.FONT_HERSHEY_TRIPLEX, 1, font_color, 2, cv2.LINE_AA)

    cv2.imshow("announce", announce)
    key = cv2.waitKey()
    cv2.destroyWindow("announce")

    if key == ord("u"):
        announce_u = np.full((100, 1300, 3), background_color, np.uint8)
        text = "Make sure the excel file is in the same directory and press any key"
        cv2.putText(announce_u, text, (50, 50), cv2.FONT_HERSHEY_TRIPLEX, 1, font_color, 2, cv2.LINE_AA)
        cv2.imshow("announce", announce_u)
        cv2.waitKey()

        light_magnitude = model.read_spectroscope(file_directory='20211024.xlsx', lower_bound=510, upper_bound=540)

    elif key == ord("d"):
        announce_d = np.full((200, 1300, 3), background_color, np.uint8)
        text = "Press f if this is the first time you check dust in your mask"
        cv2.putText(announce_d, text, (50, 50), cv2.FONT_HERSHEY_TRIPLEX, 1, font_color, 2, cv2.LINE_AA)
        text = "Press s if this is the second time you check dust in your mask"
        cv2.putText(announce_d, text, (50, 100), cv2.FONT_HERSHEY_TRIPLEX, 1, font_color, 2, cv2.LINE_AA)
        cv2.imshow("announce", announce_d)
        text = "After press f or s, Enter dust info in console"
        cv2.putText(announce_d, text, (50, 150), cv2.FONT_HERSHEY_TRIPLEX, 1, font_color, 2, cv2.LINE_AA)
        cv2.imshow("announce", announce_d)
        key_d = cv2.waitKey()

        dust_info = float(input("Enter dust info: \n"))

        print("go back to the window")

        if key_d == ord("f"):
            dust_before = dust_info
        elif key_d == ord("s"):
            dust_after = dust_info

    elif key == ord("k"):
        announce_k = np.full((250, 1500, 3), background_color, np.uint8)
        text = "Make sure riboflavin and dust have been measured and press any key"
        cv2.putText(announce_k, text, (50, 50), cv2.FONT_HERSHEY_TRIPLEX, 1, font_color, 2, cv2.LINE_AA)
        text = "(dust has to be measured twice)"
        cv2.putText(announce_k, text, (50, 100), cv2.FONT_HERSHEY_TRIPLEX, 1, font_color, 2, cv2.LINE_AA)
        text = "(If there is no measurement, press f)"
        cv2.putText(announce_k, text, (50, 150), cv2.FONT_HERSHEY_TRIPLEX, 1, font_color, 2, cv2.LINE_AA)
        cv2.imshow("announce", announce_k)
        key_k = cv2.waitKey()
        if key_k == ord("f"):
            continue
        dust_variance = dust_before - dust_after
        data = np.array([light_magnitude, dust_variance])
        model.weighted_KNN(K=5, 
                           data=data, 
                           reference=[reference_pos, reference_neg], 
                           weight=np.array([rivo_weight, dust_weight]))
        model.visualize()

    elif key == ord("w"):
        dust_list = list(model.get_dust_info())
        if dust_list[0] == "좋음":
            dust_list[0] = "good"
        elif dust_list[0] == "보통":
            dust_list[0] = "so so"
        else:
            dust_list[0] = "bad"

        if dust_list[1] == "좋음":
            dust_list[1] = "good"
        elif dust_list[1] == "보통":
            dust_list[1] = "so so"
        else:
            dust_list[1] = "bad"

        announce_k = np.full((200, 1000, 3), background_color, np.uint8)
        text = "fine dust: " + dust_list[0] + ", " + "ultra fine dust: " + dust_list[1]
        cv2.putText(announce_k, text, (50, 50), cv2.FONT_HERSHEY_TRIPLEX, 1, font_color, 2, cv2.LINE_AA)
        cv2.imshow("announce", announce_k)
        cv2.waitKey()
        
    elif key == ord("q"):
        break

    else:
        err_announce = np.full((100, 700, 3), background_color, np.uint8)
        err_text = "wrong input!! press any key"
        cv2.putText(err_announce, err_text, (50, 50), cv2.FONT_HERSHEY_TRIPLEX, 1, font_color, 2, cv2.LINE_AA)

        cv2.imshow("err_announce", err_announce)
        cv2.waitKey()
